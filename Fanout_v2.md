# IR 侧补齐 Profiling Fanout 的方案

## 一、问题回顾

- **Runtime 侧(已修完)**:fanin 写入 consumer 自己的 payload,设备端读取本任务 payload 生成 fanin 记录,host 端反转为 fanout。此路径解决了 `wire_task` / `early_finished` 的时序 race。
- **剩余缺口**:`tensormap_and_ringbuffer` 的 **ring buffer slot 回收**。当 consumer 被 submit 时,已 `CONSUMED` 且推进过 `last_task_alive` 的 producer slot_state 已被回收,orchestrator 的 `tensor.output_map` 无法再反查到它们。实测 64 个 producer 只能看到最后 16 个(加一个 dispatch 任务共 17),剩余 48 条边彻底丢失。
- **结论**:在 ring buffer 运行态上拿到全部 64 条边,等于放弃流水式回收,代价不可接受。**应从 IR 侧取静态依赖图补齐丢失边。**

## 二、核心观察

单次运行的 `build_output/ReproFanoutLost_20260421_162516/` 已经包含所有所需信息:

1. **`orchestration/repro.cpp`**:最终生成的 orchestration C++,按**词法顺序**列出所有 `pto2_rt_submit_<core>_task(func_id, params)` 调用。每次调用对应一个 runtime task,submit 顺序就是 task 的**局部索引**(ring 内 local_id)。
2. **`passes_dump/29_after_Simplify.py`**:最终 IR。orchestration 函数里仍是干净的 Python 风格,保留了 `pl.parallel(64)` 循环和单个 consumer 调用,以及每个 call 读/写的张量切片。
3. **依赖语义**:`tensor.slice` + 函数 inout/return 暴露的张量写入关系,等价于 IR 层的 producer-consumer 图。对本例:64 个 producer 写 `tmp__ssa_v0[i,:]`,1 个 consumer 读 `tmp__ssa_v0` → 64 条边。

由于 orchestrator submit 顺序 = IR 中 CallExpr 的遍历顺序 = 生成 C++ 中调用顺序,**IR task 与 runtime task 之间存在确定的顺序映射**,可以无歧义地用 `(func_id, submit_index)` 对齐。

## 三、方案总览

增加一个**编译期任务依赖导出**,host 端 profiling 合并运行时 fanin 与 IR 静态 fanout,最终在 swimlane JSON 里用**两个字段**表达两条信息:

- `fanout`:runtime 实际记录到的边(现有字段,保留)
- `fanout_ir`:IR 静态推导出的完整边集(新字段)

swimlane_converter.py 新增 `--edge-mode {runtime, ir, union, diff}`,默认 `union`,把 IR 边并入可视化。

## 四、实现分三阶段

### 阶段 A — 最小可用 POC(1-2 个文件,解决本用例)

**目标**:在 build_output 目录里生成一份静态 fanout 清单,host 端读取并注入 JSON。不碰编译器内部。

#### A1. 编译期 dump 任务图(Python 侧)

在现有 orchestration cpp 生成点附近(grep `pto2_rt_submit_aiv_task` 找到生成器),同时输出一份 JSON sidecar:

**`build_output/<run>/orchestration/task_graph.json`**
```json
{
  "tasks": [
    {"submit_index": 0, "func_id": 0, "writes": ["tmp__ssa_v0[0,:]"]},
    {"submit_index": 1, "func_id": 0, "writes": ["tmp__ssa_v0[1,:]"]},
    ...
    {"submit_index": 63, "func_id": 0, "writes": ["tmp__ssa_v0[63,:]"]},
    {"submit_index": 64, "func_id": 1, "reads": ["tmp__ssa_v0"]}
  ],
  "edges": [
    {"from_submit_index": 0, "to_submit_index": 64},
    {"from_submit_index": 1, "to_submit_index": 64},
    ...
    {"from_submit_index": 63, "to_submit_index": 64}
  ]
}
```

边通过"读张量覆盖谁的写切片"推导:对每个 consumer 的读张量,扫所有之前 submit 的 producer 写切片,看切片是否与读范围有交叠。切片信息在 IR 里就是 `tensor.slice` 的 offsets/shape。

#### A2. Host 端合并并注入 JSON

**`runtime/src/a2a3/platform/src/host/performance_collector.cpp`** 新增 `LoadIRTaskGraph(build_output_dir)`,在写 JSON 前加载 `task_graph.json`,构建 `submit_index → task_id` 的映射(需要另一个维度的信息,见下面 B1)。

**过渡方案**(不依赖 runtime 额外信息):因为 runtime task 的 `submit_index` 在同一 func_id 内是单调的,可以先按 `(func_id, per_func_submit_order)` 对齐。consumer 有 1 个(func_id=1),producer 64 个(func_id=0 的第 0..63 次)。

输出:
```json
{
  "task_id": "0x41",
  "func_id": 1,
  "fanin_count": 17,
  "fanin_actual_count_raw": 17,
  "fanout": [],
  "fanout_ir": ["0x00", "0x01", ..., "0x3f"]
}
```

每个 producer 记录里也反向填 `fanout_ir: ["0x41"]`。

#### A3. swimlane_converter.py 新增 edge-mode

**`runtime/tools/swimlane_converter.py`** 的 `fanout` 读取点(L504, L654)改为:

```python
if args.edge_mode == "runtime":
    edges = task.get("fanout", [])
elif args.edge_mode == "ir":
    edges = task.get("fanout_ir", [])
elif args.edge_mode == "union":  # default
    edges = list({*task.get("fanout", []), *task.get("fanout_ir", [])})
elif args.edge_mode == "diff":
    ir_set = set(task.get("fanout_ir", []))
    rt_set = set(task.get("fanout", []))
    edges = list(ir_set - rt_set)  # 只画 runtime 丢掉的
```

**POC 验证**:`repro_fanout_lost.py` 跑出来 Perfetto 上能看到 64/64 边。

### 阶段 B — 通用化(覆盖所有 IR/runtime)

#### B1. 编译器里新增导出 pass

**文件**:新建 `src/ir/transforms/export_task_graph.cpp`,配套 `include/pypto/ir/transforms/passes.h` 的 factory,注册在 pass manager 后期(`AllocateMemoryAddr` 之后,codegen 之前)。

责任:遍历 orchestration 函数,识别 `pto2_rt_submit_*_task` 对应的 CallExpr,按词法顺序累计 `submit_index`;对每个 call 收集输入/输出张量及切片范围;用已有切片交叠算法(codegen 里应该已有)建立边。写出 sidecar JSON。

复杂度 O(N log N):建索引 `tensor_name → [(submit_index, slice)]`,消费端查找即可。

#### B2. runtime 注入 `submit_index` 到 PerfRecord

`PTO2TaskPayload` 里本来就有 `submit_id`(orchestrator 分配),如果已经包含,直接在 `perf_aicpu_complete_record` 额外带上 `submit_index`;否则新增一个 u32 字段即可(PerfRecord 还有对齐空间)。

这样 host 端合并时就不需要靠 `(func_id, per_func_order)` 近似,而是 `submit_index` 严格一一对应。

#### B3. host 端反转改为读 IR 图 + runtime fanin 合并

```cpp
// 阶段 B 完整实现
auto ir_graph = LoadIRTaskGraph(build_output_dir);
// submit_index → task_id 双向索引(来自 PerfRecord)
std::unordered_map<uint32_t, uint64_t> idx_to_tid;
for (const auto& rec : all_records) idx_to_tid[rec.submit_index] = rec.task_id;

// fanout_ir 从 ir_graph.edges 映射
std::unordered_map<uint64_t, std::vector<uint64_t>> fanout_ir;
for (const auto& e : ir_graph.edges) {
    auto it_p = idx_to_tid.find(e.from);
    auto it_c = idx_to_tid.find(e.to);
    if (it_p != idx_to_tid.end() && it_c != idx_to_tid.end())
        fanout_ir[it_p->second].push_back(it_c->second);
}

// runtime fanout(已有)
std::unordered_map<uint64_t, std::vector<uint64_t>> fanout_rt;
for (const auto& rec : all_records)
    for (int i = 0; i < rec.fanin_count; i++)
        fanout_rt[rec.fanin[i]].push_back(rec.task_id);

// 写 JSON:两个字段都输出
```

### 阶段 C — 验证与可视化

1. `repro_fanout_lost.py` 应看到 `fanout_ir` 有 64 条边、`fanout` 有 17(runtime 实测)。
2. 现有小例子(`01_elementwise.py`)上,runtime 与 IR fanout 应基本一致,`diff` 模式应为空。
3. 长跑模型例子上,`diff` 模式能直观展示 ring 回收造成的边丢失。

## 五、受影响文件(不改 a5)

| 阶段 | 文件 | 作用 |
| --- | --- | --- |
| A1 | orchestration 生成器(Python 侧) | dump `task_graph.json` sidecar |
| A2 | `runtime/src/a2a3/platform/src/host/performance_collector.cpp` | 加载 sidecar,注入 `fanout_ir` |
| A3 | `runtime/tools/swimlane_converter.py` | `--edge-mode` 选项 |
| B1 | 新 `src/ir/transforms/export_task_graph.cpp`,`include/pypto/ir/transforms/passes.h`,`python/bindings/modules/passes.cpp`,`python/pypto/pypto_core/passes.pyi` | 编译器内统一导出任务图 |
| B2 | `runtime/src/a2a3/platform/include/common/perf_profiling.h`,3 个 runtime 的 aicpu_executor.cpp | PerfRecord 增加 `submit_index`,版本 bump 到 4 |
| B3 | `performance_collector.cpp` | 基于 `submit_index` 精确合并 |

## 六、复杂度与内存

- IR 导出 pass:O(N log N)(N=任务数,建张量名 → 切片列表索引,消费端二分/map 查找)。
- Host 合并:O(E) (E=边数,典型 O(N))。
- PerfRecord 多 4 字节 `submit_index`,占用基本不变(原有对齐空隙)。
- `task_graph.json` 每次编译几百 KB 级别,和 passes_dump 同数量级。

## 七、非目标

- 不改 a5。
- 不碰 tensormap_and_ringbuffer 的 slot 回收策略(保持流水效率)。
- 不改变 runtime 侧 fanin 写入路径(已修复,继续保留)。
- swimlane JSON schema 向后兼容:`fanout` 字段仍在,新字段可选。
