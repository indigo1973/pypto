# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO runtime runner.

Provides :func:`run`, the main entry point for compiling a ``@pl.program`` and
executing it on an Ascend NPU (or simulator), with correctness validation against
a user-supplied golden function.

Typical usage::

    import torch
    from pypto.runtime import run, RunConfig, TensorSpec

    def golden(tensors, params):
        tensors["out"][:] = tensors["a"] + tensors["b"]

    result = run(
        program=MyProgram,
        tensor_specs=[
            TensorSpec("a",   [128, 128], torch.float32, init_value=2.0),
            TensorSpec("b",   [128, 128], torch.float32, init_value=3.0),
            TensorSpec("out", [128, 128], torch.float32, is_output=True),
        ],
        golden=golden,
        config=RunConfig(platform="a2a3sim"),
    )
    print(result)  # PASS / FAIL: ...
"""

import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pypto import ir
from pypto.backend import BackendType, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy

from .golden_writer import write_golden
from .tensor_spec import TensorSpec


@dataclass
class RunConfig:
    """Configuration for a :func:`run` invocation or harness test execution.

    Attributes:
        platform: Target execution platform — ``"a2a3sim"`` / ``"a2a3"``
            (Ascend 910B) or ``"a5sim"`` / ``"a5"`` (Ascend 950).
        device_id: Hardware device index (ignored for simulator).
        rtol: Relative tolerance for result comparison.
        atol: Absolute tolerance for result comparison.
        strategy: PyPTO optimisation strategy applied during compilation.
        backend_type: Code-generation backend (:attr:`BackendType.Ascend910B` by default).
        dump_passes: If ``True``, dump intermediate IR after each pass.
        save_kernels: If ``True``, retain generated artefacts after execution.
            When ``False`` (default), a temporary directory is used and cleaned up.
        save_kernels_dir: Directory to save generated artefacts when *save_kernels*
            is ``True``.  If ``None``, a timestamped directory is created under
            ``build_output/<program_name>_<timestamp>``.
        codegen_only: If ``True``, stop after code generation without executing
            on device.  Useful for validating compilation output.
        enable_profiling: If ``True``, enable runtime profiling and generate
            ``swimlane.json`` after execution.
    """

    __test__ = False  # Not a pytest test class

    platform: str = "a2a3sim"
    device_id: int = 0
    rtol: float = 1e-5
    atol: float = 1e-5
    strategy: OptimizationStrategy = field(default_factory=lambda: OptimizationStrategy.Default)
    backend_type: BackendType = field(default_factory=lambda: BackendType.Ascend910B)
    dump_passes: bool = False
    save_kernels: bool = False
    save_kernels_dir: str | None = None
    codegen_only: bool = False
    enable_profiling: bool = False

    def __post_init__(self) -> None:
        if self.platform not in ("a2a3sim", "a2a3", "a5sim", "a5"):
            raise ValueError(
                f"Invalid platform {self.platform!r}. Expected 'a2a3sim', 'a2a3', 'a5sim', or 'a5'."
            )
        # Auto-correct platform to match backend_type so compilation and execution
        # always target the same architecture.
        expected_arch = "a5" if self.backend_type == BackendType.Ascend950 else "a2a3"
        if not self.platform.startswith(expected_arch):
            sim_suffix = "sim" if self.platform.endswith("sim") else ""
            self.platform = f"{expected_arch}{sim_suffix}"


@dataclass
class RunResult:
    """Result of a program run or harness test execution.

    Attributes:
        passed: ``True`` if the program executed and results matched the golden
            reference within the configured tolerances.
        test_name: Optional test case name.  Set by the harness when running
            a named test case; ``None`` for direct :func:`run` calls.
        error: Human-readable error message when ``passed`` is ``False``.
        execution_time: Wall-clock time in seconds for the full run (compile +
            execute + validate).
    """

    __test__ = False  # Not a pytest test class

    passed: bool
    test_name: str | None = None
    error: str | None = None
    execution_time: float | None = None

    def __str__(self) -> str:
        time_str = f" ({self.execution_time:.2f}s)" if self.execution_time else ""
        if self.passed:
            prefix = f"PASS: {self.test_name}" if self.test_name else "PASS"
            return prefix + time_str
        if self.test_name:
            msg = f"FAIL: {self.test_name}"
            if self.error:
                msg += f" - {self.error}"
        else:
            msg = "FAIL"
            if self.error:
                msg += f": {self.error}"
        return msg + time_str


def compile_program(
    program: Any,
    work_dir: Path,
    *,
    strategy: OptimizationStrategy,
    backend_type: BackendType,
    dump_passes: bool = False,
) -> None:
    """Compile *program* to *work_dir* and patch orchestration headers.

    Runs :func:`ir.compile` then inserts ``runtime.h`` / ``<iostream>`` includes
    into the generated orchestration C++ files (required by Simpler's CodeRunner).

    Args:
        program: A ``@pl.program`` decorated class or an ``ir.Program`` object.
        work_dir: Output directory for generated artefacts.
        strategy: PyPTO optimisation strategy applied during compilation.
        backend_type: Code-generation backend.
        dump_passes: If ``True``, dump intermediate IR after each pass.
    """
    ir.compile(
        program,
        output_dir=str(work_dir),
        strategy=strategy,
        dump_passes=dump_passes,
        backend_type=backend_type,
    )
    _patch_orchestration_headers(work_dir)


def run(
    program: Any,
    tensor_specs: list[TensorSpec],
    golden: Callable,
    config: RunConfig | None = None,
) -> RunResult:
    """Compile *program* and run it on device, validating against *golden*.

    The full pipeline executed by this function:

    1. Call :func:`ir.compile` to generate CCE C++ kernel and orchestration files.
    2. Patch the orchestration file with the required ``runtime.h`` header.
    3. Write a ``golden.py`` file from *tensor_specs* and *golden*.
    4. Invoke Simpler's ``CodeRunner`` to compile, load, execute, and validate.

    Args:
        program: A ``@pl.program`` decorated class or an ``ir.Program`` object.
        tensor_specs: Ordered list of tensor specifications.  The order must match
            the parameter order of the program's orchestration function.
        golden: A function with signature ``golden(tensors, params)`` that
            computes the expected outputs in-place (writes to
            ``tensors[output_name]``).  The function name does not matter.
        config: Run configuration.  Uses default :class:`RunConfig` if ``None``.

    Returns:
        :class:`RunResult` with ``passed=True`` on success, or ``passed=False``
        with an ``error`` message on failure.

    Example:
        >>> result = run(MyProgram, specs, my_golden, RunConfig(platform="a2a3sim"))
        >>> assert result.passed, str(result)
    """
    if config is None:
        config = RunConfig()

    start_time = time.time()
    if config.save_kernels_dir:
        work_dir = Path(config.save_kernels_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = Path("build_output") / f"{program.name}_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Set backend for code generation
        set_backend_type(config.backend_type)

        # 2. Compile: generates kernels/, orchestration/, kernel_config.py
        #    and patches orchestration headers
        compile_program(
            program,
            work_dir,
            strategy=config.strategy,
            backend_type=config.backend_type,
            dump_passes=config.dump_passes,
        )

        # 3. Write golden.py
        golden_path = work_dir / "golden.py"
        write_golden(tensor_specs, golden, golden_path, rtol=config.rtol, atol=config.atol)

        # 4. Execute via Simpler's CodeRunner
        _execute_on_device(work_dir, golden_path, config.platform, config.device_id,
                           config.enable_profiling)

        return RunResult(passed=True, execution_time=time.time() - start_time)

    except Exception:
        return RunResult(
            passed=False,
            error=traceback.format_exc(),
            execution_time=time.time() - start_time,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _execute_on_device(
    work_dir: Path,
    golden_path: Path,
    platform: str,
    device_id: int,
    enable_profiling: bool = False,
) -> None:
    """Invoke Simpler's CodeRunner to compile, load, execute, and validate.

    Automatically adds SIMPLER_ROOT sub-paths to ``sys.path`` when the
    ``SIMPLER_ROOT`` environment variable is set (mirrors conftest.py behaviour).

    Args:
        work_dir: Root output directory produced by :func:`compile_program`,
            containing ``kernels/`` and ``orchestration/``.
        golden_path: Path to the generated ``golden.py`` file.
        platform: Target execution platform (``"a2a3sim"``, ``"a2a3"``,
            ``"a5sim"``, or ``"a5"``).
        device_id: Hardware device index.
        enable_profiling: If ``True``, enable runtime profiling and generate
            ``swimlane.json`` after execution.
    """
    simpler_root = os.environ.get("SIMPLER_ROOT")
    if simpler_root:
        for sub in ("examples/scripts", "python"):
            p = str(Path(simpler_root) / sub)
            if p not in sys.path:
                sys.path.insert(0, p)

    from code_runner import CodeRunner  # type: ignore[import]  # noqa: PLC0415,I001 — available after sys.path setup

    # Snapshot existing device logs before run so we can identify the new one
    # (CANN writes device logs asynchronously after execution).
    pre_run_logs: set[Path] = set()
    device_log_dir: Path | None = None
    if enable_profiling:
        device_log_dir = _get_device_log_dir(device_id)
        if device_log_dir.exists():
            pre_run_logs = set(device_log_dir.glob("*.log"))

    CodeRunner(
        kernels_dir=str(work_dir),
        golden_path=str(golden_path),
        platform=platform,
        device_id=device_id,
        clone_protocol="https",
        enable_profiling=enable_profiling,
    ).run()

    if enable_profiling:
        _generate_swimlane(work_dir, device_id, device_log_dir, pre_run_logs, simpler_root)


def _get_device_log_dir(device_id: int) -> Path:
    """Return the CANN device log directory for *device_id*."""
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        root = Path(ascend_work_path).expanduser() / "log" / "debug"
        if root.exists():
            return root / f"device-{device_id}"
    return Path.home() / "ascend" / "log" / "debug" / f"device-{device_id}"


def _wait_for_new_device_log(
    log_dir: Path, pre_run_logs: set[Path], timeout: float = 15, interval: float = 0.5
) -> Path | None:
    """Wait for a new ``*.log`` file in *log_dir* that wasn't present before the run.

    CANN dlog writes device logs asynchronously, so the file may appear
    a few seconds after execution completes.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_dir.exists():
            new_logs = set(log_dir.glob("*.log")) - pre_run_logs
            if new_logs:
                return max(new_logs, key=lambda p: p.stat().st_mtime)
        time.sleep(interval)
    return None


def _generate_swimlane(
    work_dir: Path,
    device_id: int,
    device_log_dir: Path | None,
    pre_run_logs: set[Path],
    simpler_root: str | None,
) -> None:
    """Run Simpler's swimlane_converter.py to generate ``merged_swimlane_*.json``.

    Output is written to ``<project_root>/outputs/`` and named
    ``merged_swimlane_<timestamp>.json``.  Failures are logged as warnings so
    they do not cause the test to fail.

    Args:
        work_dir: Directory containing ``kernel_config.py``.
        device_id: Hardware device index (fallback when no device log found).
        device_log_dir: CANN device log directory snapshotted before the run.
        pre_run_logs: Set of log files that existed before the run.
        simpler_root: Path to the Simpler repository root.
    """
    import subprocess
    import warnings
    # Project root (python/pypto/runtime/ → python/pypto/ → python/ → project root)
    _OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent.parent/"outputs"

    if not simpler_root:
        return

    swimlane_script = Path(simpler_root) / "tools" / "swimlane_converter.py"
    if not swimlane_script.exists():
        warnings.warn(f"swimlane_converter.py not found at {swimlane_script}, skipping swimlane generation")
        return

    kernel_config_path = work_dir / "kernel_config.py"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _OUTPUTS_DIR / f"merged_swimlane_{timestamp}.json"

    cmd = [
        sys.executable, str(swimlane_script),
        "-k", str(kernel_config_path),
        "-o", str(output_path),
    ]

    if device_log_dir is not None:
        device_log_file = _wait_for_new_device_log(device_log_dir, pre_run_logs)
        if device_log_file:
            cmd += ["--device-log", str(device_log_file)]
        else:
            cmd += ["-d", str(device_id)]
    else:
        cmd += ["-d", str(device_id)]

    try:
        subprocess.run(cmd, check=True)
        print(f"Swimlane JSON written to: {output_path}")
    except subprocess.CalledProcessError as e:
        warnings.warn(f"swimlane_converter.py failed (exit {e.returncode}), no swimlane generated")

def _patch_orchestration_headers(work_dir: Path) -> None:
    """Add ``runtime.h`` and ``<iostream>`` includes to orchestration C++ files.

    Simpler's CodeRunner requires these headers in the orchestration translation
    unit.  They are added here rather than in the code generator so that the
    compiler back-end remains unaware of runtime-specific requirements.

    Args:
        work_dir: Root output directory produced by :func:`ir.compile`.
    """
    orch_dir = work_dir / "orchestration"
    if not orch_dir.exists():
        return
    for cpp_file in orch_dir.glob("*.cpp"):
        _add_headers_to_file(cpp_file)


def _add_headers_to_file(cpp_file: Path) -> None:
    """Insert missing ``runtime.h`` / ``<iostream>`` headers into *cpp_file*.

    Args:
        cpp_file: Path to a C++ source file that may be missing the headers.
    """
    content = cpp_file.read_text(encoding="utf-8")

    has_runtime_h = '#include "runtime.h"' in content
    has_iostream = "#include <iostream>" in content

    if has_runtime_h and has_iostream:
        return  # Nothing to do

    headers: list[str] = []
    if not has_runtime_h:
        headers.append('#include "runtime.h"')
    if not has_iostream:
        headers.append("#include <iostream>")

    # Find the first non-comment, non-blank line as the insertion point.
    lines = content.splitlines(keepends=True)
    insert_pos = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("//", "/*", "*")):
            insert_pos = i
            break

    header_block = "\n".join(headers) + "\n"
    if insert_pos > 0:
        header_block += "\n"

    lines.insert(insert_pos, header_block)
    cpp_file.write_text("".join(lines), encoding="utf-8")
