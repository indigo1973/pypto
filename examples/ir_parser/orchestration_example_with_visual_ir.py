# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Example Orchestration Function with Visual IR Export

This script extends orchestration_example.py to also export Visual IR JSON.
"""

import os

import pypto.language as pl
from pypto import DataType, ir
from pypto.backend import BackendType


@pl.program
class ExampleOrchProgram:
    """Example orchestration program with InCore kernels."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Adds two tensors element-wise: result = a + b"""
        a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
        result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
        output_new: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
        return output_new

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add_scalar(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        scalar: pl.Scalar[pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Adds a scalar to each element: result = a + scalar"""
        x: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        result: pl.Tile[[16, 16], pl.FP32] = pl.add(x, scalar)
        output_new: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
        return output_new

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_mul(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Multiplies two tensors element-wise: result = a * b"""
        a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
        result: pl.Tile[[16, 16], pl.FP32] = pl.mul(a_tile, b_tile)
        output_new: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
        return output_new

    @pl.function(type=pl.FunctionType.Orchestration)
    def BuildExampleGraph(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Build BuildExampleGraph orchestration function.

        Orchestration function for formula: f = (a + b + 1)(a + b + 2)
        """
        c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)
        d: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add_scalar(c, 1.0)  # type: ignore[reportArgumentType]
        e: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add_scalar(c, 2.0)  # type: ignore[reportArgumentType]
        f_result: pl.Tensor[[16, 16], pl.FP32] = self.kernel_mul(d, e)
        return f_result


def custom_attributes_filter(details: dict) -> dict:
    """Custom filter to select which details to show in attributes.

    This example shows only data_kind, data_type, and shape in the graph view.
    You can customize this function to show different fields.

    Args:
        details: Full details dictionary

    Returns:
        Filtered attributes dictionary
    """
    # Only include these fields in the visual graph
    selected_fields = ["data_kind", "data_type", "shape"]
    return {k: v for k, v in details.items() if k in selected_fields}


def main():
    """Main function - complete compilation workflow with Visual IR export."""
    print("=" * 70)
    print("Example Orch Code Generation with Visual IR Export")
    print("=" * 70)

    # Configuration
    dtype = DataType.FP32
    print(f"\nConfiguration: {dtype}")

    # Step 1: Build IR
    print("\n[1] Building IR...")
    program = ExampleOrchProgram
    import pdb
    pdb.set_trace()
    print("✓ IR construction complete")
    print(f"  Functions: {[f.name for f in program.functions.values()]}")

    # Step 2: Export Visual IR JSON (default - all details in attributes)
    print("\n[2] Exporting Visual IR JSON (default attributes)...")
    visual_ir_path_default = "visual_ir_output_default.json"
    ir.export_to_visual_ir(
        program,
        visual_ir_path_default,
        version="1.0",
        entry_function="BuildExampleGraph",
    )
    print(f"✓ Visual IR exported to: {visual_ir_path_default}")

    # Step 3: Export Visual IR JSON (custom attributes filter)
    print("\n[3] Exporting Visual IR JSON (custom attributes)...")
    visual_ir_path_custom = "visual_ir_output_custom.json"
    ir.export_to_visual_ir(
        program,
        visual_ir_path_custom,
        version="1.0",
        entry_function="BuildExampleGraph",
        attributes_filter=custom_attributes_filter,
    )
    print(f"✓ Visual IR exported to: {visual_ir_path_custom}")
    print("  Custom filter: showing only data_kind, data_type, and shape")

    # Step 4: Print IR preview
    print("\n[4] IR Preview (Python syntax):")
    print("-" * 70)
    ir_text = ir.python_print(program)
    lines = ir_text.split("\n")
    preview_lines = min(40, len(lines))
    print("\n".join(lines[:preview_lines]))
    if len(lines) > preview_lines:
        print(f"\n... ({len(lines) - preview_lines} more lines)")
    print("-" * 70)

    # Step 5: Compile (using high-level ir.compile API)
    print("\n[5] Compiling with PassManager and CCECodegen...")
    output_dir = ir.compile(
        program,
        strategy=ir.OptimizationStrategy.Default,
        dump_passes=True,
        backend_type=BackendType.CCE,
    )
    print("✓ Compilation complete")
    print(f"✓ Output directory: {output_dir}")

    # Step 6: Display generated files
    print("\n[6] Generated files:")
    for root, _dirs, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, output_dir)
            file_size = os.path.getsize(filepath)
            print(f"  - {rel_path} ({file_size} bytes)")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"  Program: {program.name}")
    print(f"  Functions: {len(program.functions)}")
    print("    - kernel_add (InCore)")
    print("    - kernel_add_scalar (InCore)")
    print("    - kernel_mul (InCore)")
    print("    - BuildExampleGraph (Orchestration)")
    print(f"  Visual IR (default): {visual_ir_path_default}")
    print(f"  Visual IR (custom): {visual_ir_path_custom}")
    print(f"  Compiled output: {output_dir}")
    print(f"  Data type: {dtype}")
    print("  Formula: f = (a + b + 1)(a + b + 2)")
    print("=" * 70)


if __name__ == "__main__":
    main()
