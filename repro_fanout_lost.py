"""Reproducer: runtime swimlane trace drops fanout edges when a producer
completes before its consumer is wired (the `early_finished` fast path in the
scheduler).
"""
import argparse
import json
import sys
from pathlib import Path

N_PRODUCERS = 64
ROW = 64


def build_program():
    import pypto.language as pl

    @pl.program
    class ReproFanoutLost:
        @pl.function(type=pl.FunctionType.Opaque)
        def repro(
            self,
            a: pl.Tensor[[N_PRODUCERS, ROW], pl.FP32],
            out: pl.Out[pl.Tensor[[1, ROW], pl.FP32]],
        ) -> pl.Tensor[[1, ROW], pl.FP32]:
            tmp = pl.create_tensor([N_PRODUCERS, ROW], dtype=pl.FP32)
            # Scope 1: N parallel fast producers (func id 0).
            for i in pl.parallel(N_PRODUCERS):
                with pl.at(level=pl.Level.CORE_GROUP):
                    row = pl.mul(pl.slice(a, [1, ROW], [i, 0]), 2.0)
                tmp = pl.assemble(tmp, row, [i, 0])

            # Scope 2: single consumer (func id 1) fans in from all N producers.
            with pl.at(level=pl.Level.CORE_GROUP):
                acc = pl.full([1, ROW], dtype=pl.FP32, value=0.0)
                for i in pl.range(N_PRODUCERS):
                    acc = pl.add(acc, pl.slice(tmp, [1, ROW], [i, 0]))
            out = pl.assemble(out, acc, [0, 0])
            return out

    return ReproFanoutLost


def run(platform, device, runtime_profiling):
    import torch
    from pypto import ir
    from pypto.backend import BackendType
    from pypto.runtime import RunConfig

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B
    compiled = ir.compile(build_program(), backend_type=backend, platform=platform)
    a = torch.randn(N_PRODUCERS, ROW, dtype=torch.float32)
    out = torch.zeros(1, ROW, dtype=torch.float32)
    compiled(a, out, config=RunConfig(platform=platform, device_id=device,
             backend_type=backend, runtime_profiling=runtime_profiling))


def analyze(json_path):
    data = json.load(open(json_path))
    tasks = data["tasks"]
    by_func = {}
    for t in tasks:
        by_func.setdefault(t["func_id"], []).append(t)
    for fid in sorted(by_func):
        g = by_func[fid]
        empty = sum(1 for t in g if not t.get("fanout"))
        print(f"func_id={fid}: {len(g)} tasks, {len(g) - empty} non-empty fanout, {empty} empty")
    producers, consumers = by_func.get(0, []), by_func.get(1, [])
    if len(consumers) == 1:
        cid = consumers[0]["task_id"]
        feeders = [p for p in producers if cid in (p.get("fanout") or [])]
        print(f"Producers listing consumer in fanout: {len(feeders)}/{len(producers)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--platform", default="a2a3")
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--runtime-profiling", action="store_true")
    ap.add_argument("--analyze", type=Path)
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze)
    else:
        run(args.platform, args.device, args.runtime_profiling)