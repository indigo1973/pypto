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

#include "pypto/codegen/orchestration/orchestration_codegen.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/orchestration_op_registry.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/dependency_analyzer.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

namespace {

using namespace pypto::ir;  // NOLINT(build/namespaces)

/**
 * @brief Check if an operation is a built-in IR operation (not a user-defined function)
 *
 * Built-in operations include block-level ops (block.*), tensor-level ops (tensor.*),
 * and system ops (system.*). These are handled by specialized codegen paths rather
 * than being dispatched as task graph function calls.
 */
bool IsBuiltinOp(const std::string& op_name) {
  return op_name.find("block.") == 0 || op_name.find("tensor.") == 0 || op_name.find("system.") == 0;
}

/**
 * @brief Check if an operation is a tensor-level IR operation
 *
 * Tensor operations (tensor.create, tensor.read, tensor.view, tensor.reshape)
 * are host-side operations. Only tensor.create and tensor.read require inline C++ codegen;
 * others are metadata-only and expressed through TensorType parameters.
 */
bool IsTensorOp(const std::string& op_name) { return op_name.find("tensor.") == 0; }

// Format scalar constant as C++ literal/expression for assignment to the given C++ type.
std::string FormatConstIntValue(const ConstIntPtr& c, const std::string& cpp_type) {
  int64_t v = c->value_;
  if (cpp_type == "uint8_t" || cpp_type == "uint16_t" || cpp_type == "uint32_t" || cpp_type == "uint64_t") {
    return "static_cast<" + cpp_type + ">(" + std::to_string(v) + ")";
  }
  if (cpp_type == "int8_t" || cpp_type == "int16_t" || cpp_type == "int32_t") {
    return "static_cast<" + cpp_type + ">(" + std::to_string(v) + ")";
  }
  return std::to_string(v);  // int64_t
}

std::string FormatConstFloatValue(const ConstFloatPtr& c, const std::string& cpp_type) {
  double v = c->value_;
  if (cpp_type == "float") {
    return std::to_string(static_cast<float>(v));
  }
  return std::to_string(v);  // double
}

bool FunctionReturnsTensor(const FunctionPtr& func) {
  if (func->return_types_.empty()) {
    return false;
  }
  return func->return_types_.size() == 1 && As<TensorType>(func->return_types_[0]) != nullptr;
}

bool FunctionReturnsTuple(const FunctionPtr& func) { return func->return_types_.size() > 1; }

int CountReturnTensors(const FunctionPtr& func) {
  int count = 0;
  for (const auto& rt : func->return_types_) {
    if (As<TensorType>(rt)) {
      count++;
    }
  }
  return count;
}

std::string GenerateOrchestrationSignature(const FunctionPtr& func) {
  std::string func_name = "Build";
  if (!func->name_.empty()) {
    func_name += static_cast<char>(std::toupper(static_cast<unsigned char>(func->name_[0])));
    func_name += func->name_.substr(1);
  }
  return "int " + func_name + "(Runtime* runtime, uint64_t* args, int arg_count)";
}

std::string GenerateArgumentValidationCode(const FunctionPtr& func) {
  int tensor_param_count = 0;
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      tensor_param_count++;
    }
  }

  int return_tensor_count = CountReturnTensors(func);
  int expected_arg_count = (tensor_param_count + return_tensor_count) * 2 + 1;

  std::ostringstream oss;
  oss << "    // Validate argument count\n";
  oss << "    if (arg_count < " << expected_arg_count << ") {\n";
  oss << "        std::cerr << \"Error: Expected at least " << expected_arg_count
      << " args, got \" << arg_count << std::endl;\n";
  oss << "        return -1;\n";
  oss << "    }\n\n";

  return oss.str();
}

std::pair<std::string, std::vector<std::string>> GenerateArgumentExtractionCode(
    const FunctionPtr& func, const std::vector<std::string>& return_var_names = {}) {
  std::ostringstream oss;
  std::vector<std::string> output_tensor_names;

  oss << "    // Extract arguments\n";
  int arg_idx = 0;

  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      oss << "    void* host_" << param->name_ << " = reinterpret_cast<void*>(args[" << arg_idx++ << "]);\n";
    }
  }

  // Extract host pointers for return tensors
  if (!return_var_names.empty()) {
    for (const auto& name : return_var_names) {
      oss << "    void* host_" << name << " = reinterpret_cast<void*>(args[" << arg_idx++ << "]);\n";
      output_tensor_names.push_back(name);
    }
  }

  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      oss << "    size_t size_" << param->name_ << " = static_cast<size_t>(args[" << arg_idx++ << "]);\n";
    }
  }

  // Extract sizes for return tensors
  if (!return_var_names.empty()) {
    for (const auto& name : return_var_names) {
      oss << "    size_t size_" << name << " = static_cast<size_t>(args[" << arg_idx++ << "]);\n";
    }
  }

  oss << "\n";

  return {oss.str(), output_tensor_names};
}

std::string CalculateTensorSize(const TensorTypePtr& tensor_type) {
  std::ostringstream oss;

  // Calculate total number of elements by multiplying all dimensions
  bool first = true;
  for (const auto& dim : tensor_type->shape_) {
    if (first) {
      oss << CodegenBase::GenerateExprString(dim);
      first = false;
    } else {
      oss << " * " << CodegenBase::GenerateExprString(dim);
    }
  }

  // If shape is empty, it's a scalar (1 element)
  if (first) {
    oss << "1";
  }

  // Multiply by element size in bytes
  size_t element_bits = tensor_type->dtype_.GetBit();
  size_t element_bytes = (element_bits + 7) / 8;  // Round up to nearest byte
  oss << " * " << element_bytes;

  return oss.str();
}

// Helper: resolve the TensorType for an intermediate tensor, handling both
// single-return and tuple-return cases.
TensorTypePtr GetIntermediateTensorType(
    const ProgramPtr& program, const std::map<std::string, AssignStmtPtr>& output_tensor_assigns,
    const std::map<std::string, std::pair<std::string, int>>& tuple_element_map,
    const std::string& tensor_name) {
  std::string assign_key;
  int return_index = 0;

  auto tuple_it = tuple_element_map.find(tensor_name);
  if (tuple_it != tuple_element_map.end()) {
    assign_key = tuple_it->second.first;
    return_index = tuple_it->second.second;
  } else {
    assign_key = tensor_name;
  }

  auto assign_it = output_tensor_assigns.find(assign_key);
  CHECK(assign_it != output_tensor_assigns.end()) << "Missing assignment info for tensor: " << assign_key;

  auto call = As<Call>(assign_it->second->value_);
  CHECK(call) << "Tensor assignment must be from a Call: " << assign_key;

  std::string callee_name = call->op_->name_;
  FunctionPtr callee_func = program->GetFunction(callee_name);
  CHECK(callee_func) << "Cannot find called function: " << callee_name;
  CHECK(return_index < static_cast<int>(callee_func->return_types_.size()))
      << "Return index " << return_index << " out of bounds for function " << callee_name;

  auto tensor_type = As<TensorType>(callee_func->return_types_[return_index]);
  CHECK(tensor_type) << "Function " << callee_name << " return type at index " << return_index
                     << " must be TensorType, got " << callee_func->return_types_[return_index]->TypeName();
  return tensor_type;
}

std::string GenerateDeviceMemoryAllocationCode(
    const ProgramPtr& program, const FunctionPtr& func, const std::set<std::string>& output_tensors,
    const std::vector<std::string>& return_tensor_names, const std::set<std::string>& intermediate_tensors,
    const std::map<std::string, AssignStmtPtr>& output_tensor_assigns,
    const std::map<std::string, std::pair<std::string, int>>& tuple_element_map) {
  std::ostringstream oss;

  oss << "    // Allocate device memory for parameters\n";
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      oss << "    void* dev_" << param->name_ << " = runtime->host_api.device_malloc(size_" << param->name_
          << ");\n";

      if (output_tensors.count(param->name_)) {
        oss << "    runtime->record_tensor_pair(host_" << param->name_ << ", dev_" << param->name_
            << ", size_" << param->name_ << ");\n";
      } else {
        oss << "    runtime->host_api.copy_to_device(dev_" << param->name_ << ", host_" << param->name_
            << ", size_" << param->name_ << ");\n";
      }
    }
  }

  for (const auto& name : return_tensor_names) {
    oss << "    void* dev_" << name << " = runtime->host_api.device_malloc(size_" << name << ");\n";
    oss << "    runtime->record_tensor_pair(host_" << name << ", dev_" << name << ", size_" << name << ");\n";
  }

  oss << "\n";

  if (!intermediate_tensors.empty()) {
    oss << "    // Allocate device memory for intermediate tensors\n";
    for (const auto& tensor_name : intermediate_tensors) {
      auto tensor_type =
          GetIntermediateTensorType(program, output_tensor_assigns, tuple_element_map, tensor_name);
      std::string size_expr = CalculateTensorSize(tensor_type);
      oss << "    size_t size_" << tensor_name << " = " << size_expr << ";\n";
      oss << "    void* dev_" << tensor_name << " = runtime->host_api.device_malloc(size_" << tensor_name
          << ");\n";
    }
    oss << "\n";
  }

  return oss.str();
}

std::string GenerateSingleTaskCode(const std::string& task_var, const std::vector<std::string>& task_args,
                                   const std::vector<std::string>& task_arg_cpp_types,
                                   const std::string& callee_name, int func_id, CoreType core_type,
                                   int task_counter) {
  std::ostringstream oss;

  oss << "    // Task " << task_counter << ": Call " << callee_name << "\n";
  oss << "    uint64_t args_" << task_var << "[" << task_args.size() << "];\n";
  for (size_t i = 0; i < task_args.size(); ++i) {
    const std::string& cpp_type = task_arg_cpp_types[i];
    const std::string& value = task_args[i];
    if (cpp_type == "void*") {
      oss << "    args_" << task_var << "[" << i << "] = reinterpret_cast<uint64_t>(" << value << ");\n";
    } else {
      oss << "    { union { " << cpp_type << " v; uint64_t u; } _u; _u.v = " << value << "; args_" << task_var
          << "[" << i << "] = _u.u; }\n";
    }
  }
  const std::string core_type_str = core_type == CoreType::CUBE ? "CoreType::AIC" : "CoreType::AIV";
  oss << "    int " << task_var << " = runtime->add_task(args_" << task_var << ", " << task_args.size()
      << ", " << func_id << ", " << core_type_str << ");\n\n";

  return oss.str();
}

void ValidateOrchestrationReferences(const ProgramPtr& program, const FunctionPtr& func) {
  CHECK(func->func_type_ == FunctionType::Orchestration)
      << "ValidateOrchestrationReferences should only be called on Orchestration functions";

  class FunctionCallCollector : public IRVisitor {
   public:
    std::set<std::string> called_functions_;

    void VisitExpr_(const CallPtr& call) override {
      if (!IsBuiltinOp(call->op_->name_)) {
        called_functions_.insert(call->op_->name_);
      }
      IRVisitor::VisitExpr_(call);
    }
  };

  FunctionCallCollector collector;
  collector.VisitStmt(func->body_);

  std::vector<std::string> missing_functions;
  for (const auto& func_name : collector.called_functions_) {
    if (!program->GetFunction(func_name)) {
      missing_functions.push_back(func_name);
    }
  }

  if (!missing_functions.empty()) {
    std::ostringstream oss;
    oss << "Orchestration function '" << func->name_ << "' references undefined functions. "
        << "The Program must contain all functions referenced in orchestration calls.\n"
        << "Missing functions: [";
    for (size_t i = 0; i < missing_functions.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << "'" << missing_functions[i] << "'";
    }
    oss << "]";
    throw pypto::ValueError(oss.str());
  }
}

CallPtr ExtractCallFromStmt(const StmtPtr& stmt) {
  if (!stmt) {
    return nullptr;
  }
  if (auto assign = As<AssignStmt>(stmt)) {
    return As<Call>(assign->value_);
  }
  if (auto eval = As<EvalStmt>(stmt)) {
    return As<Call>(eval->expr_);
  }
  return nullptr;
}

std::string GenerateTaskDependenciesCode(const FunctionPtr& func,
                                         const std::map<const Call*, int>& call_to_task,
                                         const std::vector<std::string>& task_vars) {
  INTERNAL_CHECK(func) << "Internal error: Cannot analyze dependencies for null function";

  ir::DependencyAnalyzer analyzer;
  auto all_dependencies = analyzer.AnalyzeDependencies(func);

  std::ostringstream oss;
  bool has_deps = false;

  for (const auto& edge : all_dependencies) {
    if (edge.type != ir::DependencyEdge::RAW) {
      continue;
    }

    auto producer_call = ExtractCallFromStmt(edge.producer);
    auto consumer_call = ExtractCallFromStmt(edge.consumer);

    if (producer_call && consumer_call && call_to_task.count(producer_call.get()) &&
        call_to_task.count(consumer_call.get())) {
      int producer_idx = call_to_task.at(producer_call.get());
      int consumer_idx = call_to_task.at(consumer_call.get());

      if (!has_deps) {
        oss << "    // Dependencies (data-flow based)\n";
        has_deps = true;
      }
      oss << "    runtime->add_successor(" << task_vars[producer_idx] << ", " << task_vars[consumer_idx]
          << ");\n";
    }
  }

  if (has_deps) {
    oss << "\n";
  }

  return oss.str();
}

int GetOrCreateFuncId(const std::string& func_name, std::map<std::string, int>* func_name_to_id,
                      int* next_func_id) {
  if (func_name_to_id->find(func_name) == func_name_to_id->end()) {
    (*func_name_to_id)[func_name] = (*next_func_id)++;
  }
  return (*func_name_to_id)[func_name];
}

}  // namespace

CoreType InferFunctionCoreType(const FunctionPtr& func) {
  const backend::Backend* backend = backend::GetBackend();
  class CoreTypeCollector : public IRVisitor {
   public:
    explicit CoreTypeCollector(const backend::Backend* backend) : backend_(backend) {}
    std::set<PipeType> pipe_types_;

    void VisitExpr_(const CallPtr& call) override {
      if (call->op_->GetPipe().has_value()) {
        pipe_types_.insert(*call->op_->GetPipe());
      } else if (backend_ != nullptr) {
        const auto* info = backend_->GetOpInfo(call->op_->name_);
        if (info) {
          pipe_types_.insert(info->pipe);
        }
      }
      IRVisitor::VisitExpr_(call);
    }

   private:
    const backend::Backend* backend_;
  };

  CoreTypeCollector collector(backend);
  collector.VisitStmt(func->body_);

  bool has_m = collector.pipe_types_.count(PipeType::M) > 0;
  bool has_v = collector.pipe_types_.count(PipeType::V) > 0;

  CHECK(!(has_m && has_v)) << "Function " << func->name_
                           << " contains both Matrix (M) and Vector (V) pipe types. "
                           << "A function can only use one core type (CUBE or VECTOR).";

  if (has_m) {
    return CoreType::CUBE;
  }
  if (has_v) {
    return CoreType::VECTOR;
  }
  return CoreType::VECTOR;
}

OrchestrationResult GenerateOrchestration(const ir::ProgramPtr& program, const ir::FunctionPtr& func) {
  using namespace pypto::ir;  // NOLINT(build/namespaces)

  CHECK(program != nullptr) << "Cannot generate orchestration for null program";
  CHECK(func != nullptr) << "Cannot generate orchestration for null function";

  ValidateOrchestrationReferences(program, func);

  std::map<std::string, int> func_name_to_id;
  std::map<std::string, CoreType> func_name_to_core_type;
  int next_func_id = 0;

  std::ostringstream oss;

  // Statement code generator for orchestration
  class OrchestrationStmtCodegen : public CodegenBase {
   public:
    explicit OrchestrationStmtCodegen(const ProgramPtr& prog, std::map<std::string, int>* func_ids,
                                      std::map<std::string, CoreType>* core_types, int* next_id)
        : program_(prog),
          func_name_to_id_(func_ids),
          func_name_to_core_type_(core_types),
          next_func_id_(next_id) {}

    // Set tuple element mapping from OrchestrationInfoCollector
    void SetTupleElementMap(const std::map<std::string, std::pair<std::string, int>>& map) {
      for (const auto& [var_name, pair] : map) {
        const auto& [tuple_var, index] = pair;
        tuple_var_to_elements_[tuple_var].emplace_back(index, var_name);
      }
      // Sort by index so output args are in correct order
      for (auto& [key, vec] : tuple_var_to_elements_) {
        std::sort(vec.begin(), vec.end());
      }
    }

    std::string GetGeneratedCode() const { return code_.str(); }

    // --- CodegenBase pure virtual implementations ---
    [[nodiscard]] std::string GetCurrentResultTarget() const override { return current_result_var_; }

    void Emit(const std::string& line) override { code_ << line; }

    std::string GetExprAsCode(const ir::ExprPtr& expr) override { return GenerateExprString(expr); }

    [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override {
      return dtype.ToCTypeString();
    }

    int64_t GetConstIntValue(const ir::ExprPtr& expr) override {
      auto ci = As<ConstInt>(expr);
      INTERNAL_CHECK(ci) << "Internal error: expected ConstInt expression";
      return ci->value_;
    }

    std::string GetVarName(const ir::VarPtr& var) override { return var->name_; }

    void VisitStmt_(const ForStmtPtr& for_stmt) override {
      // Generate C++ for loop
      std::string loop_var = for_stmt->loop_var_->name_;
      std::string start_expr = GenerateExprString(for_stmt->start_);
      std::string stop_expr = GenerateExprString(for_stmt->stop_);
      std::string step_expr = GenerateExprString(for_stmt->step_);

      // Initialize iter_args before loop
      for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
        const auto& iter_arg = for_stmt->iter_args_[i];
        const auto& return_var = for_stmt->return_vars_[i];
        std::string init_value = GenerateExprString(iter_arg->initValue_);
        code_ << std::string(indent_, ' ') << GetCppType(iter_arg->GetType()) << " " << return_var->name_
              << " = " << init_value << ";\n";
        // Map iter_arg to return_var for use inside loop
        iter_arg_to_var_[iter_arg.get()] = return_var->name_;
      }

      // Generate for loop header
      code_ << std::string(indent_, ' ') << "for (int64_t " << loop_var << " = " << start_expr << "; "
            << loop_var << " < " << stop_expr << "; " << loop_var << " += " << step_expr << ") {\n";

      indent_ += 2;

      // Set return_var names for YieldStmt handling
      auto saved_return_var_names = current_return_var_names_;
      current_return_var_names_.clear();
      for (const auto& rv : for_stmt->return_vars_) {
        current_return_var_names_.push_back(rv->name_);
      }

      // Visit loop body
      VisitStmt(for_stmt->body_);

      // Restore
      current_return_var_names_ = saved_return_var_names;

      indent_ -= 2;
      code_ << std::string(indent_, ' ') << "}\n";
    }

    void VisitStmt_(const AssignStmtPtr& assign) override {
      std::string var_name = assign->var_->name_;

      if (auto call = As<Call>(assign->value_)) {
        const std::string& op_name = call->op_->name_;

        if (IsTensorOp(op_name)) {
          // Generate inline tensor operation code
          GenerateTensorOpCode(call, var_name);
        } else if (!IsBuiltinOp(op_name)) {
          // Generate function call (task dispatch)
          GenerateFunctionCallCode(call, var_name);
        }
      } else if (As<TupleGetItemExpr>(assign->value_)) {
        // TupleGetItemExpr: no-op in orchestration codegen.
        // Device memory is already allocated; the mapping is handled via tuple_var_to_elements_.
      } else {
        // Simple assignment
        std::string value_expr = GenerateExprString(assign->value_);
        code_ << std::string(indent_, ' ') << GetCppType(assign->var_->GetType()) << " " << var_name << " = "
              << value_expr << ";\n";
      }
    }

    void VisitStmt_(const ReturnStmtPtr& ret) override {
      // Return statement handling
      if (!ret->value_.empty()) {
        std::ostringstream comment;
        comment << "// Return: ";
        for (size_t i = 0; i < ret->value_.size(); ++i) {
          if (i > 0) comment << ", ";
          comment << GenerateExprString(ret->value_[i]);
        }
        code_ << std::string(indent_, ' ') << comment.str() << "\n";
      }
    }

    void VisitStmt_(const SeqStmtsPtr& seq) override {
      for (const auto& stmt : seq->stmts_) {
        VisitStmt(stmt);
      }
    }

    void VisitStmt_(const YieldStmtPtr& yield_stmt) override {
      // Handle yield statement - update iter_args by assigning to return_vars
      for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
        std::string value_expr = GenerateExprString(yield_stmt->value_[i]);
        if (i < current_return_var_names_.size()) {
          code_ << std::string(indent_, ' ') << current_return_var_names_[i] << " = " << value_expr << ";\n";
        }
      }
    }

    void VisitStmt_(const EvalStmtPtr& eval) override {
      if (auto call = As<Call>(eval->expr_)) {
        const std::string& op_name = call->op_->name_;
        if (IsTensorOp(op_name)) {
          GenerateTensorOpCode(call, "");
        } else if (!IsBuiltinOp(op_name)) {
          GenerateFunctionCallCode(call, "");
        }
      }
    }

    const std::map<const Call*, int>& GetCallToTask() const { return call_to_task_; }
    const std::vector<std::string>& GetTaskVars() const { return task_vars_; }

   private:
    void GenerateTensorOpCode(const CallPtr& call, const std::string& result_var) {
      const std::string& op_name = call->op_->name_;
      auto& registry = OrchestrationOpRegistry::GetInstance();
      auto codegen_func = registry.Get(op_name);

      if (codegen_func.has_value()) {
        current_result_var_ = result_var;
        std::string code = (*codegen_func)(call, *this);
        // Each registered op returns complete statements (with semicolons, without indentation).
        // We indent each line here.
        std::istringstream iss(code);
        std::string line;
        while (std::getline(iss, line)) {
          if (!line.empty()) {
            code_ << std::string(indent_, ' ') << line << "\n";
          }
        }
      }
      // Metadata-only ops (tensor.view, tensor.reshape, tensor.transpose) have no codegen
    }

    void GenerateFunctionCallCode(const CallPtr& call, const std::string& result_var) {
      const std::string& callee_name = call->op_->name_;

      FunctionPtr callee_func = program_->GetFunction(callee_name);
      INTERNAL_CHECK(callee_func != nullptr)
          << "Internal error: function '" << callee_name
          << "' not found after validation. This should have been caught earlier.";
      CoreType core_type = InferFunctionCoreType(callee_func);
      (*func_name_to_core_type_)[callee_name] = core_type;

      int func_id = GetOrCreateFuncId(callee_name, func_name_to_id_, next_func_id_);

      std::vector<std::string> task_args;
      std::vector<std::string> task_arg_cpp_types;
      for (const auto& arg : call->args_) {
        std::string var_name = TryGetVarName(arg);
        if (!var_name.empty()) {
          task_args.emplace_back("dev_" + var_name);
          task_arg_cpp_types.emplace_back("void*");
        } else if (auto const_int = As<ConstInt>(arg)) {
          std::string cpp_type = const_int->dtype().ToCTypeString();
          task_arg_cpp_types.emplace_back(cpp_type);
          task_args.emplace_back(FormatConstIntValue(const_int, cpp_type));
        } else if (auto const_float = As<ConstFloat>(arg)) {
          std::string cpp_type = const_float->dtype().ToCTypeString();
          task_arg_cpp_types.emplace_back(cpp_type);
          task_args.emplace_back(FormatConstFloatValue(const_float, cpp_type));
        } else if (auto const_bool = As<ConstBool>(arg)) {
          task_arg_cpp_types.emplace_back("bool");
          task_args.emplace_back(const_bool->value_ ? "true" : "false");
        }
      }

      // Append output device pointers
      auto tuple_it = tuple_var_to_elements_.find(result_var);
      if (tuple_it != tuple_var_to_elements_.end()) {
        // Tuple return: append device pointers for each unpacked element
        for (const auto& [index, elem_var] : tuple_it->second) {
          task_args.emplace_back("dev_" + elem_var);
          task_arg_cpp_types.emplace_back("void*");
        }
      } else if (!result_var.empty()) {
        task_args.emplace_back("dev_" + result_var);
        task_arg_cpp_types.emplace_back("void*");
      }

      std::string task_var = "t" + std::to_string(task_counter_);
      task_vars_.push_back(task_var);
      call_to_task_[call.get()] = task_counter_;

      code_ << GenerateSingleTaskCode(task_var, task_args, task_arg_cpp_types, callee_name, func_id,
                                      core_type, task_counter_);

      task_counter_++;
    }

    std::string GetCppType(const TypePtr& type) {
      if (auto scalar_type = As<ScalarType>(type)) {
        return scalar_type->dtype_.ToCTypeString();
      }
      return "auto";
    }

    const ProgramPtr& program_;
    std::map<std::string, int>* func_name_to_id_;
    std::map<std::string, CoreType>* func_name_to_core_type_;
    int* next_func_id_;
    std::ostringstream code_;
    int indent_ = 4;
    std::map<const IterArg*, std::string> iter_arg_to_var_;
    std::string current_result_var_;
    std::vector<std::string> current_return_var_names_;
    int task_counter_ = 0;
    std::vector<std::string> task_vars_;
    std::map<const Call*, int> call_to_task_;
    // Tuple var -> sorted list of (index, unpacked_var_name)
    std::map<std::string, std::vector<std::pair<int, std::string>>> tuple_var_to_elements_;
  };

  // Collect metadata from IR (return_vars, output_tensors, tuple info) for memory allocation
  class OrchestrationInfoCollector : public IRVisitor {
   public:
    std::vector<std::string> return_vars;
    std::set<std::string> output_tensors;
    std::map<const Call*, std::string> call_to_result_var;
    std::map<std::string, AssignStmtPtr> output_tensor_assigns;            // Store AssignStmt for type info
    std::map<std::string, std::pair<std::string, int>> tuple_element_map;  // var -> (tuple_var, index)
    std::set<std::string> tuple_temp_vars;  // Tuple temporary variables (not real tensors)

    void VisitStmt_(const ReturnStmtPtr& ret) override {
      for (const auto& val : ret->value_) {
        std::string name = CodegenBase::TryGetVarName(val);
        if (!name.empty()) {
          return_vars.push_back(name);
        }
      }
      IRVisitor::VisitStmt_(ret);
    }

    void VisitStmt_(const AssignStmtPtr& assign) override {
      if (auto call = As<Call>(assign->value_)) {
        if (!IsBuiltinOp(call->op_->name_)) {
          std::string var_name = assign->var_->name_;

          // Check if this call returns a TupleType
          if (As<TupleType>(call->GetType())) {
            // Tuple-returning call: mark as tuple temp, don't add to output_tensors
            tuple_temp_vars.insert(var_name);
            call_to_result_var[call.get()] = var_name;
            output_tensor_assigns[var_name] = assign;
          } else {
            // Single tensor return (existing behavior)
            output_tensors.insert(var_name);
            call_to_result_var[call.get()] = var_name;
            output_tensor_assigns[var_name] = assign;
          }
        }
      } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
        // Handle: mi = TupleGetItemExpr(_tuple_tmp, 0)
        std::string var_name = assign->var_->name_;
        std::string tuple_var_name = CodegenBase::TryGetVarName(tuple_get->tuple_);

        if (!tuple_var_name.empty() && tuple_temp_vars.count(tuple_var_name)) {
          tuple_element_map[var_name] = {tuple_var_name, tuple_get->index_};
          output_tensors.insert(var_name);
          output_tensor_assigns[var_name] = assign;
        }
      }
      IRVisitor::VisitStmt_(assign);
    }
  };

  OrchestrationInfoCollector info_collector;
  info_collector.VisitStmt(func->body_);

  // Build return variable names for argument extraction
  // For single tensor return, use the return_var from IR; for tuple, use all return_vars
  const std::vector<std::string>& return_vars = info_collector.return_vars;

  oss << "#include <cstdint>\n\n";
  oss << "extern \"C\" {\n\n";
  oss << GenerateOrchestrationSignature(func) << " {\n";
  oss << GenerateArgumentValidationCode(func);

  auto [arg_extraction_code, return_tensor_names] = GenerateArgumentExtractionCode(func, return_vars);
  oss << arg_extraction_code;
  const std::set<std::string>& output_tensors = info_collector.output_tensors;

  std::set<std::string> param_names;
  for (const auto& param : func->params_) {
    param_names.insert(param->name_);
  }

  std::set<std::string> return_var_set(return_vars.begin(), return_vars.end());
  std::set<std::string> intermediate_tensors;
  for (const auto& var_name : output_tensors) {
    if (param_names.find(var_name) == param_names.end() &&
        return_var_set.find(var_name) == return_var_set.end()) {
      intermediate_tensors.insert(var_name);
    }
  }

  oss << GenerateDeviceMemoryAllocationCode(program, func, output_tensors, return_tensor_names,
                                            intermediate_tensors, info_collector.output_tensor_assigns,
                                            info_collector.tuple_element_map);

  // Use OrchestrationStmtCodegen to generate code with proper control flow (for loops, etc.)
  OrchestrationStmtCodegen stmt_codegen(program, &func_name_to_id, &func_name_to_core_type, &next_func_id);
  stmt_codegen.SetTupleElementMap(info_collector.tuple_element_map);
  stmt_codegen.VisitStmt(func->body_);
  oss << stmt_codegen.GetGeneratedCode();

  oss << GenerateTaskDependenciesCode(func, stmt_codegen.GetCallToTask(), stmt_codegen.GetTaskVars());

  oss << "    return 0;\n";
  oss << "}\n\n";
  oss << "}  // extern \"C\"\n";

  return OrchestrationResult{oss.str(), std::move(func_name_to_id), std::move(func_name_to_core_type)};
}

}  // namespace codegen
}  // namespace pypto
