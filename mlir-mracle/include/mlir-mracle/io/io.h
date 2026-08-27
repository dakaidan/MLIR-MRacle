#pragma once

#include "mlir-mracle/core/types.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mlir_mracle {

// IR printing utilities for artifact logging
std::string dumpMLIR(mlir::ModuleOp module);
std::string dumpLLVM(llvm::Module &module);

// ordered JSON value: object fields are emitted in insertion order so
// user-facing output keeps a stable, explicit field layout
struct JsonValue {
    enum class Kind { Null, Bool, Int, Double, String, Array, Object };
    Kind kind = Kind::Null;
    bool boolVal = false;
    int64_t intVal = 0;
    double doubleVal = 0;
    std::string strVal;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;
};

JsonValue jsonBool(bool v);
JsonValue jsonInt(int64_t v);
JsonValue jsonString(std::string v);
JsonValue jsonArray();
void jsonPush(JsonValue &arr, JsonValue v);
JsonValue jsonObject();
void jsonPut(JsonValue &obj, std::string key, JsonValue v);

// renders a JSON value with 2-space indentation
void printJson(const JsonValue &val, llvm::raw_ostream &os);

// serializes a single observed outcome set (baseline cache payload)
JsonValue observedOutcomeSetToJson(const ObservedOutcomeSet &set);

// parses an observed outcome set from a JSON object; returns false on
// malformed input
bool observedOutcomeSetFromJson(const llvm::json::Object &o,
                                ObservedOutcomeSet &set);

// renders an outcome-set comparison result as a JSON object
JsonValue outcomeSetResultToJson(const OutcomeSetResult &result);

// "OK", "WARN", or "ERROR" for a run
std::string runStatusString(const RunInfo &info);

// renders a run's short metadata for the campaign result.json
JsonValue runInfoToJson(const RunInfo &info);

// renders a run with thread results for the run_info.json artifact
JsonValue runInfoToStatusJson(const RunInfo &info);

// renders a default-pipeline run with union outcome sets and a concise
// per-binary breakdown for the run_info.json artifact
JsonValue runInfoToUnionJson(const RunInfo &info);

// renders an execution-mode run with grouped thread results for result.json
JsonValue executionRunToJson(const ExecutionRunResult &run);

} // namespace mlir_mracle
