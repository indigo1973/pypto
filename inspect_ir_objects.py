#!/usr/bin/env python3
"""
PyPTO IR 对象属性查看工具

用法:
    python inspect_ir_objects.py
"""

import sys
sys.path.insert(0, 'python')

from examples.ir_parser.orchestration_example_with_visual_ir import ExampleOrchProgram


def inspect_object(obj, name="Object", indent=0):
    """递归查看对象的属性"""
    prefix = "  " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  类型: {type(obj).__name__}")
    print(f"{prefix}  模块: {type(obj).__module__}")

    # 获取所有非私有属性
    attrs = [a for a in dir(obj) if not a.startswith('_')]

    # 分离属性和方法
    properties = []
    methods = []
    for attr in attrs:
        try:
            value = getattr(obj, attr)
            if callable(value):
                methods.append(attr)
            else:
                properties.append((attr, value))
        except Exception:
            pass

    # 打印属性
    if properties:
        print(f"{prefix}  属性:")
        for attr, value in properties:
            value_type = type(value).__name__
            if isinstance(value, (str, int, float, bool, type(None))):
                print(f"{prefix}    {attr}: {value} ({value_type})")
            elif isinstance(value, list):
                if len(value) <= 3:
                    print(f"{prefix}    {attr}: {value} ({value_type}, 长度: {len(value)})")
                else:
                    print(f"{prefix}    {attr}: [...] ({value_type}, 长度: {len(value)})")
            else:
                print(f"{prefix}    {attr}: <{value_type} 对象>")

    # 打印方法
    if methods:
        print(f"{prefix}  方法: {', '.join(methods)}")

    print()


def main():
    program = ExampleOrchProgram

    print("=" * 70)
    print("PyPTO IR 对象属性查看")
    print("=" * 70)
    print()

    # 1. Program
    print("1. Program 对象")
    print("-" * 70)
    inspect_object(program, "Program")

    # 2. GlobalVar
    print("2. GlobalVar 对象 (函数引用)")
    print("-" * 70)
    global_var = list(program.functions.keys())[0]
    inspect_object(global_var, "GlobalVar")
    print(f"  文档: {global_var.__doc__}")
    print()

    # 3. Function
    print("3. Function 对象 (函数定义)")
    print("-" * 70)
    func = list(program.functions.values())[0]
    inspect_object(func, "Function")
    print(f"  文档: {func.__doc__}")
    print()

    # 4. Var (参数)
    print("4. Var 对象 (变量/参数)")
    print("-" * 70)
    param = func.params[0]
    inspect_object(param, "Var (参数)")

    # 5. Type
    if param.type:
        print("5. TensorType 对象 (类型)")
        print("-" * 70)
        inspect_object(param.type, "TensorType")

    # 6. Stmt (语句)
    if func.body:
        print("6. Stmt 对象 (语句)")
        print("-" * 70)
        inspect_object(func.body, "SeqStmts (语句序列)")

    # 7. 查看所有函数
    print("7. 所有函数列表")
    print("-" * 70)
    for i, (gv, f) in enumerate(program.functions.items(), 1):
        print(f"  {i}. {f.name}")
        print(f"     GlobalVar: {gv.name}")
        print(f"     Function: {f.name}")
        print(f"     类型: {f.func_type}")
        print(f"     参数: {[p.name for p in f.params]}")
        print(f"     返回类型数量: {len(f.return_types)}")
        print()


if __name__ == "__main__":
    main()
