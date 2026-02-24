# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Visual IR JSON Exporter

This module exports PyPTO IR to Visual IR JSON format conforming to visual-ir_schema.json.
"""

import json
from typing import Any, Callable, Optional

from pypto.pypto_core import ir as _ir


class VisualIRExporter:
    """Exports PyPTO IR to Visual IR JSON format."""

    def __init__(
        self,
        attributes_filter: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ):
        """Initialize the Visual IR exporter.

        Args:
            attributes_filter: Optional function to customize which details fields
                             should be included in the attributes section.
                             If None, all details fields are copied to attributes.
        """
        self.attributes_filter = attributes_filter
        self.node_id_counter = 0
        self.edge_id_counter = 0
        self.node_map: dict[int, int] = {}  # IR node id -> visual IR node id
        self.outcast_params: dict[int, _ir.Var] = {}  # IR node id -> OUTCAST parameter Var

    def export_program(
        self,
        program: _ir.Program,
        version: str = "1.0",
        entry_function: Optional[str] = None,
    ) -> dict[str, Any]:
        """Export a PyPTO Program to Visual IR JSON format.

        Args:
            program: The PyPTO Program to export
            version: Visual IR format version
            entry_function: Name of the entry function (defaults to first function)

        Returns:
            Dictionary conforming to visual-ir_schema.json
        """
        # Initialize counters for global unique IDs across all graphs
        self.node_id_counter = 0
        self.edge_id_counter = 0

        # Determine entry function
        if entry_function is None:
            if len(program.functions) > 0:
                entry_function = list(program.functions.keys())[0]
            else:
                entry_function = "main"

        # Build metadata
        metadata = {
            "entry_id": entry_function,
            "name": program.name,
            "description": f"Visual IR for program {program.name}",
        }

        # Build graphs for each function
        graphs = []
        for func_name, func in program.functions.items():
            graph = self._export_function(func)
            graphs.append(graph)

        return {
            "version": version,
            "metadata": metadata,
            "graphs": graphs,
        }

    def _export_function(self, func: _ir.Function) -> dict[str, Any]:
        """Export a single function to a graph structure.

        Args:
            func: The function to export

        Returns:
            Graph dictionary with nodes and edges
        """
        # Note: Do NOT reset counters here to maintain global unique IDs across graphs
        # This enables cross-graph navigation and references
        self.node_map.clear()
        self.outcast_params.clear()

        nodes = []
        edges = []

        # Check if this is an InCore function
        is_incore = self._is_incore_function(func)

        # Analyze parameter usage to determine INCAST vs OUTCAST (only for InCore functions)
        if is_incore:
            param_roles = self._analyze_parameter_roles(func)
        else:
            # For non-InCore functions (Orchestration), all parameters are DATA
            param_roles = {id(param): "DATA" for param in func.params}

        # Add parameter nodes with appropriate roles
        # For InCore functions, skip OUTCAST parameters (they will be represented by return values)
        for param in func.params:
            role = param_roles.get(id(param), "INCAST")

            # Skip OUTCAST parameters in InCore functions (they're redundant with return values)
            # Store them for later reference in block.store operations
            if is_incore and role == "OUTCAST":
                self.outcast_params[id(param)] = param
                continue

            node = self._create_node_from_var(param, role=role)
            nodes.append(node)

        # Process function body
        if func.body:
            self._process_stmt(func.body, nodes, edges)

        # Mark return values as OUTCAST (only for InCore functions)
        if is_incore:
            self._mark_return_values_as_outcast(func, nodes)

        return {
            "id": func.name,
            "nodes": nodes,
            "edges": edges,
        }

    def _is_incore_function(self, func: _ir.Function) -> bool:
        """Check if a function is an InCore function.

        InCore functions contain block-level operations (load/store).
        Orchestration functions call other functions but don't have block ops.

        Args:
            func: The function to check

        Returns:
            True if the function is InCore
        """
        # Check if function body contains block.load or block.store operations
        if not func.body:
            return False

        return self._contains_block_operations(func.body)

    def _contains_block_operations(self, stmt: _ir.Stmt) -> bool:
        """Check if a statement contains block-level operations.

        Args:
            stmt: The statement to check

        Returns:
            True if block operations are found
        """
        stmt_type = type(stmt).__name__

        if stmt_type == "AssignStmt":
            return self._expr_is_block_operation(stmt.value)
        elif stmt_type == "SeqStmts":
            return any(self._contains_block_operations(s) for s in stmt.stmts)
        elif stmt_type == "IfStmt":
            result = False
            if stmt.then_body:
                result = result or self._contains_block_operations(stmt.then_body)
            if stmt.else_body:
                result = result or self._contains_block_operations(stmt.else_body)
            return result
        elif stmt_type == "ForStmt":
            if stmt.body:
                return self._contains_block_operations(stmt.body)
        elif stmt_type == "EvalStmt":
            return self._expr_is_block_operation(stmt.value)

        return False

    def _expr_is_block_operation(self, expr: _ir.Expr) -> bool:
        """Check if an expression is a block-level operation.

        Args:
            expr: The expression to check

        Returns:
            True if this is a block operation
        """
        expr_type = type(expr).__name__

        if expr_type == "Call" and hasattr(expr, 'op'):
            op = expr.op
            op_name = ""
            if hasattr(op, 'name'):
                op_name = op.name
            elif hasattr(op, 'name_'):
                op_name = op.name_

            # Check for block operations
            if op_name.startswith("block."):
                return True

        return False

    def _analyze_parameter_roles(self, func: _ir.Function) -> dict[int, str]:
        """Analyze function parameters to determine INCAST vs OUTCAST roles.

        Parameters used in block.load operations are INCAST (data movement in).
        Parameters used in block.store operations are OUTCAST (data movement out).
        Other parameters are DATA (no data movement).

        Args:
            func: The function to analyze

        Returns:
            Dictionary mapping parameter id to role ("INCAST", "OUTCAST", or "DATA")
        """
        param_roles: dict[int, str] = {}

        # Initialize all parameters as DATA by default
        for param in func.params:
            param_roles[id(param)] = "DATA"

        # Analyze function body to find load/store operations
        if func.body:
            self._analyze_stmt_for_roles(func.body, param_roles)

        return param_roles

    def _analyze_stmt_for_roles(
        self, stmt: _ir.Stmt, param_roles: dict[int, str]
    ) -> None:
        """Recursively analyze statements to identify parameter roles.

        Args:
            stmt: The statement to analyze
            param_roles: Dictionary to update with parameter roles
        """
        stmt_type = type(stmt).__name__

        if stmt_type == "AssignStmt":
            # Check if this is a store operation
            self._analyze_expr_for_roles(stmt.value, param_roles)
        elif stmt_type == "SeqStmts":
            for s in stmt.stmts:
                self._analyze_stmt_for_roles(s, param_roles)
        elif stmt_type == "IfStmt":
            if stmt.then_body:
                self._analyze_stmt_for_roles(stmt.then_body, param_roles)
            if stmt.else_body:
                self._analyze_stmt_for_roles(stmt.else_body, param_roles)
        elif stmt_type == "ForStmt":
            if stmt.body:
                self._analyze_stmt_for_roles(stmt.body, param_roles)

    def _analyze_expr_for_roles(
        self, expr: _ir.Expr, param_roles: dict[int, str]
    ) -> None:
        """Analyze expressions to identify parameters used in load/store operations.

        Args:
            expr: The expression to analyze
            param_roles: Dictionary to update with parameter roles
        """
        expr_type = type(expr).__name__

        if expr_type == "Call":
            # Check if this is a load or store operation
            if hasattr(expr, 'op'):
                op = expr.op
                op_name = ""
                if hasattr(op, 'name'):
                    op_name = op.name
                elif hasattr(op, 'name_'):
                    op_name = op.name_

                # block.load has signature: load(source_tensor, offset, shape)
                # The first argument (source_tensor) is the INCAST parameter
                if op_name == "block.load" and hasattr(expr, 'args'):
                    args = expr.args
                    if len(args) >= 1:
                        # The 1st argument (index 0) is the source tensor
                        src_arg = args[0]
                        if type(src_arg).__name__ == "Var":
                            # Mark this parameter as INCAST
                            param_roles[id(src_arg)] = "INCAST"

                # block.store has signature: store(data, offset, shape, dest_tensor)
                # The last argument (dest_tensor) is the OUTCAST parameter
                elif op_name == "block.store" and hasattr(expr, 'args'):
                    args = expr.args
                    if len(args) >= 4:
                        # The 4th argument (index 3) is the destination tensor
                        dest_arg = args[3]
                        if type(dest_arg).__name__ == "Var":
                            # Mark this parameter as OUTCAST
                            param_roles[id(dest_arg)] = "OUTCAST"

    def _mark_return_values_as_outcast(
        self, func: _ir.Function, nodes: list[dict[str, Any]]
    ) -> None:
        """Mark return value nodes as OUTCAST.

        Args:
            func: The function being exported
            nodes: List of nodes to update
        """
        # Find return statement
        if not func.body:
            return

        return_vars = self._find_return_vars(func.body)

        # Update role for return value nodes
        for node in nodes:
            node_name = node.get("name", "")
            if any(var.name == node_name for var in return_vars):
                node["role"] = "OUTCAST"

    def _find_return_vars(self, stmt: _ir.Stmt) -> list[_ir.Var]:
        """Find variables returned by the function.

        Args:
            stmt: The statement to search

        Returns:
            List of Var nodes that are returned
        """
        result = []
        stmt_type = type(stmt).__name__

        if stmt_type == "ReturnStmt":
            if hasattr(stmt, 'value') and stmt.value:
                if isinstance(stmt.value, list):
                    for val in stmt.value:
                        if type(val).__name__ == "Var":
                            result.append(val)
                elif type(stmt.value).__name__ == "Var":
                    result.append(stmt.value)
        elif stmt_type == "SeqStmts":
            for s in stmt.stmts:
                result.extend(self._find_return_vars(s))
        elif stmt_type == "IfStmt":
            if stmt.then_body:
                result.extend(self._find_return_vars(stmt.then_body))
            if stmt.else_body:
                result.extend(self._find_return_vars(stmt.else_body))
        elif stmt_type == "ForStmt":
            if stmt.body:
                result.extend(self._find_return_vars(stmt.body))

        return result


    def _create_node_from_var(
        self, var: _ir.Var, role: str = "DATA"
    ) -> dict[str, Any]:
        """Create a node from a Var IR node.

        Args:
            var: The Var node
            role: Node role (DATA, OP, INCAST, OUTCAST, DEFAULT)

        Returns:
            Node dictionary
        """
        node_id = self.node_id_counter
        self.node_id_counter += 1
        self.node_map[id(var)] = node_id

        details = self._extract_details_from_var(var)
        attributes = self._filter_attributes(details)

        return {
            "id": node_id,
            "name": var.name,
            "role": role,
            "attributes": attributes,
            "details": details,
        }

    def _create_node_from_expr(
        self, expr: _ir.Expr, role: str = "DATA", name: Optional[str] = None,
        input_ids: Optional[list[int]] = None, output_id: Optional[int] = None
    ) -> dict[str, Any]:
        """Create a node from an expression.

        Args:
            expr: The expression node
            role: Node role
            name: Optional node name (auto-generated if None)
            input_ids: Optional list of input node IDs (for OP nodes)
            output_id: Optional output node ID (for OP nodes)

        Returns:
            Node dictionary
        """
        node_id = self.node_id_counter
        self.node_id_counter += 1
        self.node_map[id(expr)] = node_id

        if name is None:
            # Generate name based on expression type
            expr_type = type(expr).__name__

            # For constants, use a descriptive name without node_id
            if expr_type in ["ConstInt", "ConstFloat", "ConstBool", "ConstString"]:
                if hasattr(expr, 'value'):
                    value = expr.value
                    if expr_type == "ConstFloat":
                        name = f"const_{value}"
                    elif expr_type == "ConstInt":
                        name = f"const_{value}"
                    elif expr_type == "ConstBool":
                        name = f"const_{str(value).lower()}"
                    elif expr_type == "ConstString":
                        name = f"const_\"{value}\""
                    else:
                        name = f"const_{value}"
                else:
                    name = f"{expr_type}_{node_id}"
            else:
                # For other expressions, use type name with node_id
                name = f"{expr_type}_{node_id}"

        details = self._extract_details_from_expr(expr, role, input_ids, output_id)
        attributes = self._filter_attributes(details)

        return {
            "id": node_id,
            "name": name,
            "role": role,
            "attributes": attributes,
            "details": details,
        }

    def _extract_details_from_var(self, var: _ir.Var) -> dict[str, Any]:
        """Extract details from a Var node.

        Args:
            var: The Var node

        Returns:
            Details dictionary
        """
        details: dict[str, Any] = {}

        # Extract type information
        if var.type:
            self._add_type_details(var.type, details)

        # Extract span information
        if var.span:
            details["span_file"] = var.span.filename
            details["span_line"] = var.span.begin_line

        return details

    def _extract_details_from_expr(
        self, expr: _ir.Expr, role: str = "DATA",
        input_ids: Optional[list[int]] = None, output_id: Optional[int] = None
    ) -> dict[str, Any]:
        """Extract details from an expression node.

        Args:
            expr: The expression node
            role: Node role (determines which details to extract)
            input_ids: Optional list of input node IDs (for OP nodes)
            output_id: Optional output node ID (for OP nodes)

        Returns:
            Details dictionary
        """
        details: dict[str, Any] = {}

        # For OP nodes, add operation-specific details
        if role == "OP":
            if input_ids is not None:
                details["inputs"] = input_ids
            if output_id is not None:
                details["outputs"] = [output_id]

            # Check if this is a Call to a function (subgraph)
            if type(expr).__name__ == "Call" and hasattr(expr, 'op'):
                op = expr.op
                # If op is a GlobalVar, it references another function (subgraph)
                if type(op).__name__ == "GlobalVar":
                    # Add subgraph_id using the function name (which is the graph id)
                    if hasattr(op, 'name'):
                        details["subgraph_id"] = op.name
                    elif hasattr(op, 'name_'):
                        details["subgraph_id"] = op.name_
        else:
            # For DATA nodes, add type information
            if expr.type:
                self._add_type_details(expr.type, details)

            # For constant expressions, add the constant value
            expr_type = type(expr).__name__
            if expr_type in ["ConstInt", "ConstFloat", "ConstBool", "ConstString"]:
                if hasattr(expr, 'value'):
                    details["value"] = expr.value

        # Extract span information for all nodes
        if expr.span:
            details["span_file"] = expr.span.filename
            details["span_line"] = expr.span.begin_line

        return details

    def _add_type_details(self, type_obj: _ir.Type, details: dict[str, Any]) -> None:
        """Add type information to details dictionary.

        Args:
            type_obj: The type object
            details: Details dictionary to update
        """
        type_name = type(type_obj).__name__

        if type_name == "TensorType":
            tensor_type = type_obj
            details["data_kind"] = "tensor"
            details["data_type"] = str(tensor_type.dtype)
            details["shape"] = self._convert_shape_to_json(tensor_type.shape)
        elif type_name == "TileType":
            tile_type = type_obj
            details["data_kind"] = "tile"
            details["data_type"] = str(tile_type.dtype)
            details["shape"] = self._convert_shape_to_json(tile_type.shape)
        elif type_name == "ScalarType":
            scalar_type = type_obj
            details["data_kind"] = "scalar"
            details["data_type"] = str(scalar_type.dtype)
        else:
            details["data_kind"] = "others"

    def _convert_shape_to_json(self, shape: list) -> list:
        """Convert shape list to JSON-serializable format.

        Args:
            shape: Shape list that may contain IR expressions

        Returns:
            JSON-serializable list
        """
        result = []
        for dim in shape:
            if isinstance(dim, int):
                result.append(dim)
            elif dim is None:
                result.append(None)
            elif hasattr(dim, 'value'):
                # ConstInt or similar
                result.append(dim.value)
            else:
                # Convert to string for symbolic expressions
                result.append(str(dim))
        return result

    def _filter_attributes(self, details: dict[str, Any]) -> dict[str, Any]:
        """Filter details to create attributes dictionary.

        Args:
            details: Full details dictionary

        Returns:
            Filtered attributes dictionary
        """
        if self.attributes_filter:
            return self.attributes_filter(details)
        # Default: copy all details to attributes
        return details.copy()

    def _process_stmt(
        self, stmt: _ir.Stmt, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Process a statement and add nodes/edges.

        Args:
            stmt: The statement to process
            nodes: List to append nodes to
            edges: List to append edges to
        """
        stmt_type = type(stmt).__name__

        if stmt_type == "AssignStmt":
            self._process_assign_stmt(stmt, nodes, edges)
        elif stmt_type == "SeqStmts":
            for s in stmt.stmts:
                self._process_stmt(s, nodes, edges)
        elif stmt_type == "IfStmt":
            self._process_if_stmt(stmt, nodes, edges)
        elif stmt_type == "ForStmt":
            self._process_for_stmt(stmt, nodes, edges)
        elif stmt_type == "ReturnStmt":
            self._process_return_stmt(stmt, nodes, edges)
        elif stmt_type == "EvalStmt":
            self._process_eval_stmt(stmt, nodes, edges)

    def _process_assign_stmt(
        self, stmt: _ir.AssignStmt, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Process an assignment statement.

        Args:
            stmt: The assignment statement
            nodes: List to append nodes to
            edges: List to append edges to
        """
        # Create node for the variable being assigned
        var_node = self._create_node_from_var(stmt.var, role="DATA")
        nodes.append(var_node)

        # Process the value expression
        self._process_expr(stmt.value, nodes, edges, target_node_id=var_node["id"])

    def _process_expr(
        self,
        expr: _ir.Expr,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        target_node_id: Optional[int] = None,
        parent_op_name: Optional[str] = None,
        arg_index: Optional[int] = None,
    ) -> Optional[int]:
        """Process an expression and return its node ID.

        Args:
            expr: The expression to process
            nodes: List to append nodes to
            edges: List to append edges to
            target_node_id: Optional target node to connect to
            parent_op_name: Name of the parent operation (for context)
            arg_index: Index of this argument in parent operation

        Returns:
            Node ID of the expression, or None if node was skipped
        """
        expr_type = type(expr).__name__

        if expr_type == "Var":
            # Check if this is an OUTCAST parameter in a block.store operation
            if parent_op_name == "block.store" and id(expr) in self.outcast_params:
                # This is the destination tensor parameter - don't create a node
                # Return None to indicate this argument should be skipped
                return None

            # Variable reference - should already exist
            if id(expr) in self.node_map:
                node_id = self.node_map[id(expr)]
            else:
                # Create new node for this variable
                node = self._create_node_from_var(expr, role="DATA")
                nodes.append(node)
                node_id = node["id"]

            if target_node_id is not None:
                self._add_edge(node_id, target_node_id, edges)

            return node_id

        elif expr_type == "Call":
            # Function call - create OP node
            # Get the operation name
            op_name = "call"
            if hasattr(expr, 'op'):
                op = expr.op
                if hasattr(op, 'name'):
                    op_name = op.name
                elif hasattr(op, 'name_'):
                    op_name = op.name_
                else:
                    op_name = str(op)

            # Process arguments - skip MakeTuple and other auxiliary operations
            input_ids = []
            op_attributes = {}

            if hasattr(expr, 'args'):
                for i, arg in enumerate(expr.args):
                    # Check if this argument is a MakeTuple (should be extracted as attribute)
                    if type(arg).__name__ == "MakeTuple":
                        # Extract tuple values as operation attribute
                        tuple_values = self._extract_tuple_values(arg)
                        if tuple_values is not None:
                            # Store as attribute (e.g., offset, shape)
                            if op_name == "block.load":
                                if i == 1:
                                    op_attributes["offset"] = tuple_values
                                elif i == 2:
                                    op_attributes["shape"] = tuple_values
                            elif op_name == "block.store":
                                if i == 1:
                                    op_attributes["offset"] = tuple_values
                                elif i == 2:
                                    op_attributes["shape"] = tuple_values
                        # Don't create node or edge for MakeTuple
                        continue

                    # Check if this is an OUTCAST parameter in block.store
                    if op_name == "block.store" and type(arg).__name__ == "Var" and id(arg) in self.outcast_params:
                        # This is the destination tensor - add its info to attributes
                        outcast_var = self.outcast_params[id(arg)]
                        op_attributes["destination"] = outcast_var.name
                        # Add type information if available
                        if outcast_var.type:
                            dest_details = {}
                            self._add_type_details(outcast_var.type, dest_details)
                            op_attributes["destination_type"] = dest_details
                        # Don't create node or edge for OUTCAST parameter
                        continue

                    # Process normal arguments (Var, constants, etc.)
                    arg_node_id = self._process_expr(arg, nodes, edges, parent_op_name=op_name, arg_index=i)
                    if arg_node_id is not None:
                        input_ids.append(arg_node_id)

            # Create OP node with input/output information and attributes
            op_node = self._create_node_from_expr(
                expr, role="OP", name=op_name,
                input_ids=input_ids,
                output_id=target_node_id
            )

            # Add operation-specific attributes
            if op_attributes:
                op_node["details"].update(op_attributes)
                op_node["attributes"].update(op_attributes)

            nodes.append(op_node)

            # Create edges from inputs to OP
            for input_id in input_ids:
                self._add_edge(input_id, op_node["id"], edges)

            # Connect to target if specified
            if target_node_id is not None:
                self._add_edge(op_node["id"], target_node_id, edges)

            return op_node["id"]

        else:
            # Other expressions (constants, binary ops, etc.)
            # Check if this is a constant expression
            if self._is_constant_expr(expr):
                # Constants are DATA nodes
                node = self._create_node_from_expr(expr, role="DATA")
                nodes.append(node)

                if target_node_id is not None:
                    self._add_edge(node["id"], target_node_id, edges)

                return node["id"]
            else:
                # Binary/unary operations are OP nodes
                input_ids = []

                # Try to extract operands for binary/unary operations
                if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
                    # Binary operation
                    lhs_id = self._process_expr(expr.lhs, nodes, edges)
                    rhs_id = self._process_expr(expr.rhs, nodes, edges)
                    input_ids = [lhs_id, rhs_id]
                elif hasattr(expr, 'operand'):
                    # Unary operation
                    operand_id = self._process_expr(expr.operand, nodes, edges)
                    input_ids = [operand_id]

                node = self._create_node_from_expr(
                    expr, role="OP",
                    input_ids=input_ids if input_ids else None,
                    output_id=target_node_id
                )
                nodes.append(node)

                # Create edges from inputs to this node
                for input_id in input_ids:
                    self._add_edge(input_id, node["id"], edges)

                if target_node_id is not None:
                    self._add_edge(node["id"], target_node_id, edges)

                return node["id"]

    def _is_constant_expr(self, expr: _ir.Expr) -> bool:
        """Check if an expression is a constant.

        Args:
            expr: The expression to check

        Returns:
            True if the expression is a constant
        """
        expr_type = type(expr).__name__
        # Common constant types in PyPTO IR
        constant_types = [
            "ConstInt",
            "ConstFloat",
            "ConstBool",
            "ConstString",
        ]
        return expr_type in constant_types

    def _extract_tuple_values(self, tuple_expr: _ir.Expr) -> Optional[list[Any]]:
        """Extract values from a MakeTuple expression.

        Args:
            tuple_expr: The MakeTuple expression

        Returns:
            List of values, or None if extraction fails
        """
        if not hasattr(tuple_expr, 'elements'):
            return None

        values = []
        for element in tuple_expr.elements:
            element_type = type(element).__name__

            if element_type in ["ConstInt", "ConstFloat"]:
                if hasattr(element, 'value'):
                    values.append(element.value)
                else:
                    values.append(str(element))
            elif element_type == "Var":
                if hasattr(element, 'name'):
                    values.append(element.name)
                else:
                    values.append(str(element))
            else:
                # For other types, convert to string
                values.append(str(element))

        return values if values else None

    def _process_if_stmt(
        self, stmt: _ir.IfStmt, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Process an if statement."""
        # Process condition
        self._process_expr(stmt.condition, nodes, edges)

        # Process then branch
        if stmt.then_body:
            self._process_stmt(stmt.then_body, nodes, edges)

        # Process else branch
        if stmt.else_body:
            self._process_stmt(stmt.else_body, nodes, edges)

    def _process_for_stmt(
        self, stmt: _ir.ForStmt, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Process a for loop statement."""
        # Process loop body
        if stmt.body:
            self._process_stmt(stmt.body, nodes, edges)

    def _process_return_stmt(
        self, stmt: _ir.ReturnStmt, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Process a return statement."""
        if hasattr(stmt, 'value') and stmt.value:
            # ReturnStmt.value is a list of expressions
            if isinstance(stmt.value, list):
                for val in stmt.value:
                    self._process_expr(val, nodes, edges)
            else:
                self._process_expr(stmt.value, nodes, edges)

    def _process_eval_stmt(
        self, stmt: _ir.EvalStmt, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Process an eval statement."""
        self._process_expr(stmt.value, nodes, edges)

    def _add_edge(
        self, source_id: int, target_id: int, edges: list[dict[str, Any]], name: Optional[str] = None
    ) -> None:
        """Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edges: List to append edge to
            name: Optional edge name
        """
        edge_id = self.edge_id_counter
        self.edge_id_counter += 1

        edge: dict[str, Any] = {
            "id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
        }

        if name:
            edge["attributes"] = {"name": name}

        edges.append(edge)


def export_to_visual_ir(
    program: _ir.Program,
    output_path: str,
    version: str = "1.0",
    entry_function: Optional[str] = None,
    attributes_filter: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    indent: int = 2,
) -> None:
    """Export a PyPTO Program to Visual IR JSON file.

    Args:
        program: The PyPTO Program to export
        output_path: Path to write the JSON file
        version: Visual IR format version
        entry_function: Name of the entry function (defaults to first function)
        attributes_filter: Optional function to customize attributes from details
        indent: JSON indentation level (default: 2)

    Example:
        >>> import pypto.language as pl
        >>> from pypto import ir
        >>>
        >>> @pl.program
        >>> class MyProgram:
        >>>     @pl.function
        >>>     def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        >>>         return x
        >>>
        >>> # Export with default attributes (all details)
        >>> ir.export_to_visual_ir(MyProgram, "output.json")
        >>>
        >>> # Export with custom attributes filter
        >>> def my_filter(details):
        >>>     return {k: v for k, v in details.items() if k in ["data_kind", "shape"]}
        >>> ir.export_to_visual_ir(MyProgram, "output.json", attributes_filter=my_filter)
    """
    exporter = VisualIRExporter(attributes_filter=attributes_filter)
    visual_ir = exporter.export_program(program, version=version, entry_function=entry_function)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(visual_ir, f, indent=indent, ensure_ascii=False)

