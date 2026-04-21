# IR-Side Fanout Recovery: 详细设计方案

## 问题背景

Swimlane profiler 丢失大部分依赖边。在 64 producer → 1 consumer 场景中，只有约 16/64 条 fanout 边被记录。

**根因：** Producer 完成时遍历 `slot_state.fanout_head` 链表来记录 fanout，但 consumer 此时还未被 `wire_task` 挂到该链表上（orchestrator 还在慢慢 submit）。等 consumer 真正 wire 时，producer 的 PerfRecord 早已 seal。

**约束：** 不允许增加运行时开销。

## 方案概述

不修改运行时数据路径（零运行时开销），利用编译器已知的 IR 信息在 host 侧离线重建依赖边。

```
编译时                          运行时                         后处理
┌──────────────┐          ┌──────────────────┐         ┌──────────────────┐
│ IR 遍历每个   │          │ PerfRecord 携带   │         │ Host 侧按        │
│ submit 调用点 │──json──→ │ (task_id, func_id)│──合并──→│ func_id 分组排序  │
│ 记录 tensor   │          │ 正常采集，无改动   │         │ 匹配 IR site     │
│ root + 方向   │          └──────────────────┘         │ 推导 fanout 边   │
└──────────────┘                                        └──────────────────┘
  task_graph.json                                         fanout_ir[]
```

## 一、编译时：IR 静态分析

### 输入

`GenerateOrchestration(program, func)` 接收的 `ir::Program` 是 pass pipeline 的最终输出（等价于 `passes_dump/29_after_Simplify.py` 的内容），包含完整的函数定义和类型信息。

### TaskGraphExporter 的工作

`TaskGraphExporter`（`IRVisitor` 子类）遍历 orchestration 函数体，对每个 submit 调用点提取三个信息：

| 信息 | 来源 | 说明 |
|------|------|------|
| func_id | `func_name_to_id` 映射表（codegen 已有） | 标识被调用的 kernel |
| tensor root | `BufferRootCollector`（codegen 已有） | 追踪 SSA 变量到其根 buffer |
| direction | `CallSiteDirectionResolver`（codegen 已有） | 解析每个实参的有效方向 |

以 repro 场景为例，IR 中两个 submit 调用点：

```python
# site_id=0, func_id=0 — 在 pl.parallel(64) 循环中
row__ssa_v0 = self.repro_incore_0(a__ssa_v0, i__idx_v0, ret0__out)
#                                  ↑ root="a"  ↑ scalar   ↑ root="ret0"
#                                  dir="in"    (skip)      dir="inout"

# site_id=1, func_id=1 — 循环外
acc__rv_v2 = self.repro_incore_1(tmp__ssa_v0, ret0__out_1)
#                                 ↑ root="tmp"  ↑ root="ret0"
#                                 dir="in"       dir="inout"
```

### 输出：task_graph.json

```json
{
  "version": 1,
  "orchestration": "repro",
  "sites": [
    {"site_id": 0, "callee": "repro_incore_0", "func_id": 0, "core_type": "AIV",
     "tensors": [{"root": "a", "dir": "in"}, {"root": "ret0", "dir": "inout"}]},
    {"site_id": 1, "callee": "repro_incore_1", "func_id": 1, "core_type": "AIV",
     "tensors": [{"root": "tmp", "dir": "in"}, {"root": "ret0", "dir": "inout"}]}
  ]
}
```

### 设计决策：不分析 slice 偏移

`a[i, :]` 和 `a[j, :]` 在 IR 层面是不同 slice，但我们只记录根 tensor `"a"`。这是有意的保守策略——superset 不会漏边，精确 slice 分析复杂度高且对 profiling 场景收益有限。

## 二、Python 桥接：环境变量传递

`runner.py` 在 profiling 执行前自动设置 `PYPTO_TASK_GRAPH_JSON` 环境变量指向 `work_dir/orchestration/task_graph.json`，执行后清除。运行时通过此环境变量定位 sidecar 文件。

## 三、Host 侧后处理：匹配 + 边重建

### Step 1: 加载 task_graph.json

手写的 JSON 解析器（`LoadTaskGraphSidecar`），不引入外部依赖。仅解析 `sites` 数组。

### Step 2: 匹配 IR site 到实际 PerfRecord

运行时 PerfRecord 只携带 `(task_id, func_id)`，没有 IR site 标识。匹配依赖一个事实：

> **PTO2 runtime 对同一 func_id 的 task 按 submit 顺序分配递增的 task_id。**

`task_id` 的布局是 `[ring_id(32bit) | local_id(32bit)]`，ring_id 按 submit 顺序递增。因此：

```
func_id=0 的 PerfRecords 按 task_id 排序:
  record[0]  → IR site_id=0 的第 0 次调用（loop iteration 0）
  record[1]  → IR site_id=0 的第 1 次调用（loop iteration 1）
  ...
  record[63] → IR site_id=0 的第 63 次调用（loop iteration 63）

func_id=1 的 PerfRecords 按 task_id 排序:
  record[0]  → IR site_id=1 的第 0 次调用
```

如果一个 func_id 对应多个 IR site（同一 kernel 在不同位置被调用），用 `k % site_count` 取模匹配。

### Step 3: 边重建算法

按 `start_time` 排序所有 task，模拟 tensor 的写→读依赖：

```
初始化: root_producer_entries = {}   // tensor_root → [(task_id, site_id), ...]

按 start_time 遍历每个 task:
    site = rt_to_site[task.task_id]

    // 1. 读阶段：连接到 tensor T 的所有 producer（跳过同 site_id）
    for each tensor T where dir ∈ {in, inout}:
        for each (producer_tid, producer_site_id) in root_producer_entries[T]:
            if producer_site_id ≠ site.site_id:
                ir_fanout[producer_tid].append(task.task_id)

    // 2. 写阶段：更新 producer 列表
    for each tensor T where dir == out:
        root_producer_entries[T] = [(task.task_id, site.site_id)]   // 覆盖
    for each tensor T where dir == inout:
        root_producer_entries[T].append((task.task_id, site.site_id))  // 追加
```

**跳过同 site_id 的关键原因：** `pl.parallel(64)` 的 64 个 task 都映射到 site_id=0，它们对 `ret0` 的写入是不同 slice，不构成真实依赖。如果不跳过会产生 O(N²) 条假边。

### Step 4: 输出

每个 task 新增 `fanout_ir[]` 和 `fanout_ir_count` 字段，与原有 `fanout[]` 并列。JSON 版本号升至 4。

## 四、Swimlane 可视化合并

`swimlane_converter.py` 在生成 Chrome Trace JSON 前，将 `fanout_ir` 合并到 `fanout`：

```python
for task in tasks:
    ir_fanout = task.get("fanout_ir")
    if ir_fanout:
        merged = sorted(set(task["fanout"]) | set(ir_fanout))
        task["fanout"] = merged
        task["fanout_count"] = len(merged)
```

下游所有 flow 箭头渲染逻辑无需修改。`merged_swimlane_*.json` 中可直接看到完整的依赖箭头。

## 涉及文件

| 层 | 文件 | 仓 | 修改内容 |
|---|------|---|---------|
| Codegen header | `include/pypto/codegen/orchestration/orchestration_codegen.h` | pypto | OrchestrationResult 新增 task_graph_json 字段 |
| Codegen impl | `src/codegen/orchestration/orchestration_codegen.cpp` | pypto | TaskGraphExporter + SerializeTaskGraph + 接入 GenerateOrchestration |
| Nanobind | `python/bindings/modules/codegen.cpp` | pypto | 暴露 task_graph_json 属性 |
| Type stub | `python/pypto/pypto_core/codegen.pyi` | pypto | task_graph_json: str |
| Backend | `python/pypto/backend/pto_backend.py` | pypto | 写 task_graph.json 到 result_files |
| Runner | `python/pypto/runtime/runner.py` | pypto | 设置/清除 PYPTO_TASK_GRAPH_JSON 环境变量 |
| Host collector | `runtime/src/a2a3/platform/src/host/performance_collector.cpp` | runtime | 加载 sidecar + 匹配 + 边重建 + 输出 fanout_ir |
| Converter | `runtime/tools/swimlane_converter.py` | runtime | 合并 fanout_ir 到 fanout |

## 开销分析

### 设备侧运行时

**零。** 不修改 PTO2TaskPayload、scheduler、orchestrator、wiring queue 的任何逻辑。

### 编译时

| 项目 | 开销 |
|------|------|
| TaskGraphExporter 遍历 | O(S × P)，S=submit site 数，P=每 site 参数数 |
| JSON 序列化 | O(S × P)，字符串拼接 |
| task_graph.json 体积 | ~200 字节/site |

可忽略。编译时间主体是 pass pipeline 和 ptoas，多一次 IR 遍历占比 < 0.1%。TaskGraphExporter 没有引入新的分析能力，只是把三个已有 collector 的结果组合序列化。

### Host 侧后处理

仅在 `export_swimlane_json()` 中增加，发生在设备执行完毕、收集完 PerfRecord 之后：

| 步骤 | 复杂度 | repro 场景（N=65, S=2） |
|------|--------|------------------------|
| 加载 JSON | O(F)，F=文件大小 | 400B，μs 级 |
| 按 func_id 分组+排序 | O(N log N) | μs 级 |
| 匹配 IR site | O(N) | 65 次 map 查找 |
| 边重建遍历 | O(N × T × avg_producers) | T=每 site tensor 数，same-site 过滤后实际边数远小于 N² |
| 去重排序 | O(E log E)，E=总边数 | E=64 |
| JSON 输出 | O(N + E) | 多写 fanout_ir 字段 |

repro 场景实测整体增加 < 1ms。

### 磁盘 I/O

| 项目 | 增量 |
|------|------|
| task_graph.json | ~200B/site，编译时一次写 |
| perf_swimlane_*.json 增大 | 每个 task 多一个 fanout_ir 数组，约 +20% |

### 内存

所有数据结构（`IrSite` 向量、`rt_to_site` map、`root_producer_entries`、`ir_fanout`）均为 `export_swimlane_json()` 的局部变量，函数返回即释放，不影响常驻内存。

## 使用方法

1. 正常编译并运行（开启 `runtime_profiling=True`），环境变量自动设置
2. 生成的 `perf_swimlane_*.json` 中 `version=4`，包含 `fanout_ir[]` 字段
3. `merged_swimlane_*.json` 中依赖箭头已包含 IR 恢复的边
4. 手动使用时设置 `export PYPTO_TASK_GRAPH_JSON=/path/to/task_graph.json`

## 验证结果

repro 场景（64 producer → 1 consumer）：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| runtime fanout → consumer | 16/64 | 16/64（未变） |
| fanout_ir → consumer | N/A | **64/64** |
| merged_swimlane 箭头 | 16 条 | **64 条** |
| JSON version | 3 | 4 |
