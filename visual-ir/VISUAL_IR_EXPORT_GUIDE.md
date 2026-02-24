# Visual IR Export 使用指南

## 概述

PyPTO 现在支持将 IR 导出为 Visual IR JSON 格式，用于可视化和分析。导出的 JSON 文件符合 `visual-ir/visual-ir_schema.json` 定义的规范。

## 快速开始

### 基本用法

```python
import pypto.language as pl
from pypto import ir

@pl.program
class MyProgram:
    @pl.function
    def my_function(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        return x + 1.0

# 导出 Visual IR
ir.export_to_visual_ir(MyProgram, "output.json")
```

### 自定义 attributes 字段

默认情况下，`attributes` 字段会包含 `details` 中的所有内容。你可以通过 `attributes_filter` 参数自定义要显示的字段：

```python
def custom_filter(details: dict) -> dict:
    """只显示数据类型和形状"""
    return {
        k: v for k, v in details.items()
        if k in ["data_kind", "data_type", "shape"]
    }

ir.export_to_visual_ir(
    MyProgram,
    "output.json",
    attributes_filter=custom_filter
)
```

## API 参考

### `export_to_visual_ir()`

```python
def export_to_visual_ir(
    program: ir.Program,
    output_path: str,
    version: str = "1.0",
    entry_function: Optional[str] = None,
    attributes_filter: Optional[Callable[[dict], dict]] = None,
    indent: int = 2,
) -> None
```

**参数：**

- `program`: 要导出的 PyPTO Program 对象
- `output_path`: 输出 JSON 文件的路径
- `version`: Visual IR 格式版本（默认: "1.0"）
- `entry_function`: 入口函数名称（默认: 第一个函数）
- `attributes_filter`: 自定义 attributes 过滤函数（默认: 复制所有 details）
- `indent`: JSON 缩进级别（默认: 2）

### `VisualIRExporter` 类

如果需要更细粒度的控制，可以直接使用 `VisualIRExporter` 类：

```python
from pypto.ir import VisualIRExporter

exporter = VisualIRExporter(attributes_filter=my_filter)
visual_ir_dict = exporter.export_program(program)

# 手动保存或进一步处理
import json
with open("output.json", "w") as f:
    json.dump(visual_ir_dict, f, indent=2)
```

## 输出格式

导出的 JSON 文件包含以下结构：

```json
{
  "version": "1.0",
  "metadata": {
    "entry_id": "main",
    "name": "MyProgram",
    "description": "Visual IR for program MyProgram"
  },
  "graphs": [
    {
      "id": "function_name",
      "nodes": [
        {
          "id": 0,
          "name": "x",
          "role": "INCAST",
          "attributes": { ... },
          "details": {
            "data_kind": "tensor",
            "data_type": "FP32",
            "shape": [16],
            "span_file": "example.py",
            "span_line": 10
          }
        }
      ],
      "edges": [
        {
          "id": 0,
          "source_id": 0,
          "target_id": 1
        }
      ]
    }
  ]
}
```

## 节点角色说明

- **INCAST**: 函数输入参数
- **OUTCAST**: 函数返回值
- **DATA**: 数据节点（变量、张量等）
- **OP**: 操作节点（函数调用、运算等）
- **DEFAULT**: 默认角色

## details 字段说明

`details` 对象包含节点的详细信息。**不同角色的节点有不同的 details 字段：**

### DATA 节点和 INCAST/OUTCAST 节点（数据节点）

- `data_kind`: 数据种类（"tensor" | "scalar" | "tile" | "others"）
- `data_type`: 数据类型（如 "fp32", "int64"）
- `shape`: 数据形状数组
- `span_file`: 源代码文件位置
- `span_line`: 源代码行号

### OP 节点（操作节点）

- `inputs`: 输入节点 ID 数组
- `outputs`: 输出节点 ID 数组
- `span_file`: 源代码文件位置
- `span_line`: 源代码行号
- `subgraph_id`: 子图 ID（用于嵌套函数，可选）

**注意：** OP 节点不包含 `data_kind`、`data_type`、`shape` 等数据属性，这些属性只属于数据节点。

### 示例节点

**DATA 节点示例：**
```json
{
  "id": 2,
  "name": "c",
  "role": "DATA",
  "attributes": { "data_kind": "tensor", "data_type": "fp32", "shape": [16, 16] },
  "details": {
    "data_kind": "tensor",
    "data_type": "fp32",
    "shape": [16, 16],
    "span_file": "example.py",
    "span_line": 77
  }
}
```

**OP 节点示例：**
```json
{
  "id": 3,
  "name": "kernel_add",
  "role": "OP",
  "attributes": { "inputs": [0, 1], "outputs": [2] },
  "details": {
    "inputs": [0, 1],
    "outputs": [2],
    "span_file": "example.py",
    "span_line": 77
  }
}
```

## 示例

### 示例 1: 基本导出

参见 `examples/ir_parser/orchestration_example_with_visual_ir.py`

### 示例 2: 自定义 attributes

```python
def minimal_attributes(details: dict) -> dict:
    """只显示最基本的信息"""
    result = {}
    if "data_kind" in details:
        result["data_kind"] = details["data_kind"]
    if "shape" in details:
        result["shape"] = details["shape"]
    return result

ir.export_to_visual_ir(
    program,
    "minimal_output.json",
    attributes_filter=minimal_attributes
)
```

### 示例 3: 完整的 details

```python
# 默认行为：attributes 包含所有 details
ir.export_to_visual_ir(program, "full_output.json")
```

## 测试

运行测试脚本验证功能：

```bash
python test_visual_ir_export.py
```

## 注意事项

1. **attributes vs details**:
   - `details` 包含所有可用的节点信息
   - `attributes` 是用户选择在可视化界面中显示的字段
   - 默认情况下 `attributes` 复制所有 `details` 内容

2. **自定义过滤器**:
   - 过滤器函数接收 `details` 字典，返回 `attributes` 字典
   - 可以选择性地包含、排除或转换字段
   - 返回空字典表示不显示任何 attributes

3. **性能考虑**:
   - 对于大型程序，导出可能需要一些时间
   - 考虑只导出需要可视化的函数

## 故障排除

### 问题: 导出的 JSON 文件为空或不完整

**解决方案**: 确保 Program 对象已正确构建，包含函数和 IR 节点。

### 问题: attributes 字段为空

**解决方案**: 检查 `attributes_filter` 函数是否正确返回字典。如果返回空字典，attributes 将为空。

### 问题: 缺少某些节点或边

**解决方案**: 当前实现可能不支持所有 IR 节点类型。请报告问题以便改进。

## 未来改进

- [ ] 支持更多 IR 节点类型
- [ ] 添加节点分组和层次结构
- [ ] 支持增量导出（只导出变更部分）
- [ ] 添加可视化样式配置
- [ ] 支持导出为其他格式（GraphML, DOT 等）
