# Visual IR JSON Schema 使用说明

## 概述

本文档说明如何使用 `visual-ir_schema.json` JSON Schema 文件来验证和构建符合 PTO-IR Visual IR 规范的 JSON 文件。

## 层级结构

Visual IR JSON 文件的完整层级结构如下：

```
根对象
├── version (string, 必需) - 版本号
├── metadata (object, 可选) - 元数据
│   ├── entry_id (string, 必需) - 打开的第一张图的id，顶层图
│   ├── name (string, 可选) - 名称（目前使用program的名称）
│   ├── description (string, 可选) - 描述
│   └── ... (支持自定义扩展属性)
└── graphs (array, 必需) - 图数组
    └── graph (object, 必需) - 图结构
        ├── id (string, 必需) - 图的唯一标识（目前不是数字ID，使用function name）
        ├── nodes (array, 必需) - 节点数组
        │   └── node (object)
        │       ├── id (integer, 必需) - 节点ID
        │       ├── name (string, 必需) - 节点名称
        │       ├── role (enum, 必需) - 节点角色: "DATA" | "OP" | "DEFAULT"
        │       │   - DATA: 不涉及数据搬移的普通数据节点
        │       │   - OP: 操作节点
        │       │   - DEFAULT: 默认角色
        │       ├── attributes (object, 可选) - 用户希望显示的节点属性，支持自定义扩展
        │       │   ├── 用户选择的，希望在计算图中显示的属性字段，结构可以参考details(可选)
        │       │   └── ... (支持自定义扩展属性)
        │       └── detalils (object, 可选) - 详细的节点属性，支持自定义扩展
        │           ├── data_kind (enum, 可选) - 数据节点数据种类: "tensor" | "scalar" | "tile" | "others"
        │           ├── data_mode (enum, 可选) - 数据节点状态方式： "INCAST" | "OUTCAST" | "DEFAULT"
        │           │   - INCAST: 通过 block.load 等数据搬移操作加载的输入参数
        │           │   - OUTCAST: 通过 block.store 等数据搬移操作输出的返回值
        │           │   - DEFAULT: 默认状态
        │           ├── data_type (string, 可选) - 数据节点数据类型，如 "float16", "int8" 等
        │           ├── layout (string, 可选) - 数据节点数据格式/布局，如 "NZ", "ND", "FRACTAL_Z" 等
        │           ├── shape (array, 可选) - 数据节点数据形状数组，元素类型: number | string (scalar符号表达式，如 "%d", "%batch") | null
        │           ├── inputs (array, 可选) - 节点的所有输入的id数组
        │           ├── outputs (array, 可选) - 节点的所有输出的id数组
        │           ├── span_file (string, 可选) - 节点对应的源代码文件位置
        │           ├── span_line (integer, 可选) - 节点对应的源代码开始行数
        │           ├── subgraph_id (string, 可选) - 操作节点对应的子图ID（使用 function name）
        │           ├── offset (array, 可选) - 操作节点的偏移量数组（如 block.load/store 的 offset 参数）
        │           ├── destination (string, 可选) - 操作节点的目标参数名称（如 block.store 的目标张量名）
        │           ├── destination_type (object, 可选) - 操作节点的目标参数类型信息（包含 data_kind, data_type, shape）
        │           └── ... (支持自定义扩展属性)
        └── edges (array, 必需) - 边数组
            └── edge (object)
                ├── id (integer, 必需) - 边ID
                ├── source_id (integer, 必需) - 源节点ID
                ├── target_id (integer, 必需) - 目标节点ID
                └── attributes (object, 可选) - 用户希望显示的边属性，支持自定义扩展
                    ├── name (string, 可选) - 边名称
                    └── ... (支持自定义扩展属性)
```
