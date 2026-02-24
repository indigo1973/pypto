# PyPTO IR 表达式类型 (expr_type) 参考手册

## 概述

在 PyPTO IR 中，`expr_type` 通过 `type(expr).__name__` 获取，表示表达式的具体类型。

```python
expr_type = type(expr).__name__  # 例如: "Var", "Call", "ConstInt" 等
```

## 完整表达式类型列表

### 1. 基本表达式 (4 种)

| 类型 | 说明 | 示例 |
|------|------|------|
| `Var` | 变量引用 | `a`, `x`, `result` |
| `Call` | 函数/操作调用 | `block.load(a, [0], [16])` |
| `MakeTuple` | 元组构造 | `(0, 0)`, `[16, 16]` |
| `TupleGetItemExpr` | 元组元素访问 | `tuple[0]` |

### 2. 常量表达式 (3 种)

| 类型 | 说明 | 示例 |
|------|------|------|
| `ConstInt` | 整数常量 | `0`, `16`, `-1` |
| `ConstFloat` | 浮点数常量 | `1.0`, `3.14` |
| `ConstBool` | 布尔常量 | `True`, `False` |

### 3. 算术运算 (9 种 BinaryExpr)

| 类型 | 运算符 | 说明 |
|------|--------|------|
| `Add` | `+` | 加法 |
| `Sub` | `-` | 减法 |
| `Mul` | `*` | 乘法 |
| `FloorDiv` | `//` | 整除 |
| `FloorMod` | `%` | 取模 |
| `FloatDiv` | `/` | 浮点除法 |
| `Pow` | `**` | 幂运算 |
| `Min` | `min()` | 最小值 |
| `Max` | `max()` | 最大值 |

### 4. 比较运算 (6 种 BinaryExpr)

| 类型 | 运算符 | 说明 |
|------|--------|------|
| `Eq` | `==` | 等于 |
| `Ne` | `!=` | 不等于 |
| `Lt` | `<` | 小于 |
| `Le` | `<=` | 小于等于 |
| `Gt` | `>` | 大于 |
| `Ge` | `>=` | 大于等于 |

### 5. 逻辑运算 (3 种 BinaryExpr)

| 类型 | 运算符 | 说明 |
|------|--------|------|
| `And` | `and` | 逻辑与 |
| `Or` | `or` | 逻辑或 |
| `Xor` | `xor` | 逻辑异或 |

### 6. 位运算 (5 种 BinaryExpr)

| 类型 | 运算符 | 说明 |
|------|--------|------|
| `BitAnd` | `&` | 按位与 |
| `BitOr` | `\|` | 按位或 |
| `BitXor` | `^` | 按位异或 |
| `BitShiftLeft` | `<<` | 左移 |
| `BitShiftRight` | `>>` | 右移 |

### 7. 一元运算 (5 种 UnaryExpr)

| 类型 | 运算符 | 说明 |
|------|--------|------|
| `Abs` | `abs()` | 绝对值 |
| `Neg` | `-` | 取负 |
| `Not` | `not` | 逻辑非 |
| `BitNot` | `~` | 按位取反 |
| `Cast` | - | 类型转换 |

### 8. 其他表达式 (4 种)

| 类型 | 说明 |
|------|------|
| `BinaryExpr` | 二元运算基类 |
| `UnaryExpr` | 一元运算基类 |
| `IterArg` | 迭代参数 (for 循环) |
| `MemRef` | 内存引用 |

**总计: 39 种表达式类型**

## 在 visual_ir_exporter.py 中的使用

### 当前处理的表达式类型

```python
def _process_expr(self, expr, nodes, edges, ...):
    expr_type = type(expr).__name__

    if expr_type == "Var":
        # 处理变量引用
        ...

    elif expr_type == "Call":
        # 处理函数/操作调用
        ...

    else:
        # 处理其他表达式（常量、运算等）
        if self._is_constant_expr(expr):
            # 常量节点
            ...
        else:
            # 二元/一元运算节点
            ...
```

### 常量表达式判断

```python
def _is_constant_expr(self, expr):
    expr_type = type(expr).__name__
    constant_types = [
        "ConstInt",
        "ConstFloat",
        "ConstBool",
        "ConstString",  # 注意：实际 IR 中可能没有 ConstString
    ]
    return expr_type in constant_types
```

### 常量命名

```python
def _create_node_from_expr(self, expr, ...):
    expr_type = type(expr).__name__

    if expr_type in ["ConstInt", "ConstFloat", "ConstBool", "ConstString"]:
        if hasattr(expr, 'value'):
            value = expr.value
            if expr_type == "ConstFloat":
                name = f"const_{value}"  # const_1.0
            elif expr_type == "ConstInt":
                name = f"const_{value}"  # const_16
            elif expr_type == "ConstBool":
                name = f"const_{str(value).lower()}"  # const_true
```

## 表达式层次结构

```
Expr (基类)
├── Var
├── Call
├── MakeTuple
├── TupleGetItemExpr
├── ConstInt
├── ConstFloat
├── ConstBool
├── BinaryExpr (二元运算基类)
│   ├── Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Pow
│   ├── Min, Max
│   ├── Eq, Ne, Lt, Le, Gt, Ge
│   ├── And, Or, Xor
│   └── BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight
├── UnaryExpr (一元运算基类)
│   ├── Abs, Neg, Not, BitNot
│   └── Cast
├── IterArg
└── MemRef
```

## 实际使用示例

### 示例 1: 判断表达式类型

```python
expr_type = type(expr).__name__

if expr_type == "Var":
    print(f"变量: {expr.name}")
elif expr_type == "Call":
    print(f"调用: {expr.op.name}")
elif expr_type == "ConstInt":
    print(f"整数常量: {expr.value}")
elif expr_type == "Add":
    print("加法运算")
```

### 示例 2: 处理不同表达式

```python
def process_expression(expr):
    expr_type = type(expr).__name__

    # 基本表达式
    if expr_type == "Var":
        return f"变量 {expr.name}"

    # 常量
    if expr_type in ["ConstInt", "ConstFloat", "ConstBool"]:
        return f"常量 {expr.value}"

    # 函数调用
    if expr_type == "Call":
        op_name = expr.op.name if hasattr(expr.op, 'name') else str(expr.op)
        return f"调用 {op_name}"

    # 二元运算
    if expr_type in ["Add", "Sub", "Mul"]:
        return f"二元运算 {expr_type}"

    # 一元运算
    if expr_type in ["Neg", "Not", "Abs"]:
        return f"一元运算 {expr_type}"

    return f"其他表达式 {expr_type}"
```

### 示例 3: 在 visual_ir_exporter.py 中的实际应用

```python
# 检查 MakeTuple
if type(arg).__name__ == "MakeTuple":
    tuple_values = self._extract_tuple_values(arg)
    op_attributes["offset"] = tuple_values
    continue

# 检查常量
if expr_type in ["ConstInt", "ConstFloat", "ConstBool"]:
    node = self._create_node_from_expr(expr, role="DATA")
    nodes.append(node)

# 检查二元运算
if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
    # 这是二元运算
    lhs_id = self._process_expr(expr.lhs, nodes, edges)
    rhs_id = self._process_expr(expr.rhs, nodes, edges)
```

## 相关文件

- **类型定义**: `python/pypto/pypto_core/ir.pyi` (第 146-1229 行)
- **C++ 实现**: `include/pypto/ir/expr.h`
- **使用示例**: `python/pypto/ir/visual_ir_exporter.py`

## 注意事项

1. **使用 `type(expr).__name__` 而非 `isinstance()`**
   ```python
   # ✅ 推荐
   expr_type = type(expr).__name__
   if expr_type == "Var":
       ...

   # ❌ 不推荐（会匹配子类）
   if isinstance(expr, ir.BinaryExpr):
       ...
   ```

2. **常量类型可能不完整**
   - 当前只有 `ConstInt`, `ConstFloat`, `ConstBool`
   - `ConstString` 在代码中出现但可能未实现

3. **二元/一元运算的通用处理**
   ```python
   # 检查是否有 lhs/rhs 属性（二元运算）
   if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
       # 处理二元运算

   # 检查是否有 operand 属性（一元运算）
   elif hasattr(expr, 'operand'):
       # 处理一元运算
   ```

4. **Call 表达式的 op 属性**
   ```python
   if expr_type == "Call":
       op = expr.op
       # op 可能是 Op, GlobalVar 等类型
       op_name = op.name if hasattr(op, 'name') else str(op)
   ```
