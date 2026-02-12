/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "pypto/codegen/codegen_base.h"

#include <string>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

std::string CodegenBase::TryGetVarName(const ir::ExprPtr& expr) {
  if (auto var = As<Var>(expr)) {
    return var->name_;
  }
  if (auto iter_arg = As<IterArg>(expr)) {
    return iter_arg->name_;
  }
  return "";
}

std::string CodegenBase::GenerateExprString(const ir::ExprPtr& expr) {
  std::string var_name = TryGetVarName(expr);
  if (!var_name.empty()) {
    return var_name;
  }
  if (auto const_int = As<ConstInt>(expr)) {
    return std::to_string(const_int->value_);
  }
  if (auto add = As<Add>(expr)) {
    return "(" + GenerateExprString(add->left_) + " + " + GenerateExprString(add->right_) + ")";
  }
  if (auto sub = As<Sub>(expr)) {
    return "(" + GenerateExprString(sub->left_) + " - " + GenerateExprString(sub->right_) + ")";
  }
  if (auto mul = As<Mul>(expr)) {
    return "(" + GenerateExprString(mul->left_) + " * " + GenerateExprString(mul->right_) + ")";
  }
  if (auto floor_div = As<FloorDiv>(expr)) {
    return "(" + GenerateExprString(floor_div->left_) + " / " + GenerateExprString(floor_div->right_) + ")";
  }
  if (auto tuple_get = As<TupleGetItemExpr>(expr)) {
    return GenerateExprString(tuple_get->tuple_) + "_" + std::to_string(tuple_get->index_);
  }
  throw pypto::NotImplementedError("GenerateExprString not implemented for expression type: " +
                                   expr->TypeName());
}

}  // namespace codegen
}  // namespace pypto
