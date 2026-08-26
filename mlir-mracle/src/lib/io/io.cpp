#include "mlir-mracle/io/io.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cmath>

namespace mlir_mracle {

std::string dumpMLIR(mlir::ModuleOp module) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    module.print(os);
    os.flush();
    return buf;
}

std::string dumpLLVM(llvm::Module &module) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    module.print(os, nullptr);
    os.flush();
    return buf;
}

JsonValue jsonBool(bool b) {
    JsonValue v;
    v.kind = JsonValue::Kind::Bool;
    v.boolVal = b;
    return v;
}

JsonValue jsonInt(int64_t i) {
    JsonValue v;
    v.kind = JsonValue::Kind::Int;
    v.intVal = i;
    return v;
}

JsonValue jsonString(std::string s) {
    JsonValue v;
    v.kind = JsonValue::Kind::String;
    v.strVal = std::move(s);
    return v;
}

JsonValue jsonArray() {
    JsonValue v;
    v.kind = JsonValue::Kind::Array;
    return v;
}

void jsonPush(JsonValue &arr, JsonValue v) {
    arr.array.push_back(std::move(v));
}

JsonValue jsonObject() {
    JsonValue v;
    v.kind = JsonValue::Kind::Object;
    return v;
}

void jsonPut(JsonValue &obj, std::string key, JsonValue v) {
    obj.object.emplace_back(std::move(key), std::move(v));
}

namespace {

void printValue(const JsonValue &val, llvm::raw_ostream &os, int indent) {
    std::string pad(indent * 2, ' ');
    switch (val.kind) {
    case JsonValue::Kind::Null:
        os << "null";
        break;
    case JsonValue::Kind::Bool:
        os << (val.boolVal ? "true" : "false");
        break;
    case JsonValue::Kind::Int:
        os << val.intVal;
        break;
    case JsonValue::Kind::Double: {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", val.doubleVal);
        os << buf;
        break;
    }
    case JsonValue::Kind::String: {
        os << '"';
        for (char c : val.strVal) {
            switch (c) {
            case '"':
                os << "\\\"";
                break;
            case '\\':
                os << "\\\\";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\t':
                os << "\\t";
                break;
            case '\r':
                os << "\\r";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    os << buf;
                } else {
                    os << c;
                }
            }
        }
        os << '"';
        break;
    }
    case JsonValue::Kind::Array: {
        if (val.array.empty()) {
            os << "[]";
            break;
        }
        bool compact = true;
        for (const auto &v : val.array)
            if (v.kind == JsonValue::Kind::Array ||
                v.kind == JsonValue::Kind::Object) {
                compact = false;
                break;
            }
        if (compact) {
            os << "[";
            for (size_t i = 0; i < val.array.size(); ++i) {
                if (i > 0)
                    os << ", ";
                printValue(val.array[i], os, indent);
            }
            os << "]";
        } else {
            os << "[\n";
            for (size_t i = 0; i < val.array.size(); ++i) {
                if (i > 0)
                    os << ",\n";
                os << pad << "  ";
                printValue(val.array[i], os, indent + 1);
            }
            os << "\n" << pad << "]";
        }
        break;
    }
    case JsonValue::Kind::Object: {
        if (val.object.empty()) {
            os << "{}";
            break;
        }
        os << "{\n";
        for (size_t i = 0; i < val.object.size(); ++i) {
            if (i > 0)
                os << ",\n";
            os << pad << "  \"" << val.object[i].first << "\": ";
            printValue(val.object[i].second, os, indent + 1);
        }
        os << "\n" << pad << "}";
        break;
    }
    }
}

} // namespace

void printJson(const JsonValue &val, llvm::raw_ostream &os) {
    printValue(val, os, 0);
}

JsonValue jointOutcomeToJson(const JointOutcome &o) {
    JsonValue arr = jsonArray();
    for (int64_t v : o)
        jsonPush(arr, jsonInt(v));
    return arr;
}

JsonValue outcomeListToJson(const std::vector<JointOutcome> &outcomes,
                            const std::vector<int64_t> &counts) {
    JsonValue arr = jsonArray();
    for (size_t i = 0; i < outcomes.size(); ++i) {
        JsonValue entry = jsonObject();
        jsonPut(entry, "outcome", jointOutcomeToJson(outcomes[i]));
        jsonPut(entry, "count",
                jsonInt(i < counts.size() ? counts[i] : 1));
        jsonPush(arr, std::move(entry));
    }
    return arr;
}

JsonValue observedOutcomeSetToJson(const ObservedOutcomeSet &set) {
    JsonValue obj = jsonObject();
    jsonPut(obj, "version", jsonInt(1));
    jsonPut(obj, "outcomes", outcomeListToJson(set.outcomes, set.counts));
    jsonPut(obj, "arity", jsonInt(static_cast<int64_t>(set.arity)));
    jsonPut(obj, "arity_consistent", jsonBool(set.arityConsistent));
    jsonPut(obj, "total_runs", jsonInt(set.totalRuns));
    return obj;
}

bool observedOutcomeSetFromJson(const llvm::json::Object &o,
                                ObservedOutcomeSet &set) {
    // a version mismatch means the payload predates the current format; it is
    // treated as a cache miss rather than being silently reinterpreted
    if (auto v = o.getInteger("version"); !v || *v != 1)
        return false;
    const llvm::json::Array *arr = o.getArray("outcomes");
    if (!arr)
        return false;
    for (const auto &cv : *arr) {
        const auto *obj = cv.getAsObject();
        if (!obj)
            return false;
        const llvm::json::Array *va = obj->getArray("outcome");
        if (!va)
            return false;
        JointOutcome jo;
        for (const auto &ev : *va)
            if (auto v = ev.getAsInteger())
                jo.push_back(static_cast<int64_t>(*v));
        set.outcomes.push_back(std::move(jo));
        if (auto c = obj->getInteger("count"))
            set.counts.push_back(*c);
        else
            set.counts.push_back(1);
    }
    if (auto a = o.getInteger("arity"))
        set.arity = static_cast<size_t>(*a);
    if (auto b = o.getBoolean("arity_consistent"))
        set.arityConsistent = *b;
    if (auto t = o.getInteger("total_runs"))
        set.totalRuns = *t;
    return true;
}

JsonValue outcomeSetResultToJson(const OutcomeSetResult &result) {
    JsonValue obj = jsonObject();
    jsonPut(obj, "source_outcomes",
            outcomeListToJson(result.source.outcomes, result.source.counts));
    jsonPut(obj, "transformed_outcomes",
            outcomeListToJson(result.transformed.outcomes,
                              result.transformed.counts));
    return obj;
}

namespace {

JsonValue requestedTransformsToJson(const RunInfo &info) {
    JsonValue arr = jsonArray();
    if (info.requestedTransforms.empty())
        jsonPush(arr, jsonString("all"));
    else
        for (const auto &r : info.requestedTransforms)
            jsonPush(arr, jsonString(r));
    return arr;
}

JsonValue appliedTransformsToJson(
    const std::vector<AppliedTransformation> &applied) {
    JsonValue arr = jsonArray();
    for (const auto &at : applied) {
        JsonValue entry = jsonObject();
        jsonPut(entry, "name", jsonString(at.name));
        jsonPut(entry, "target_function", jsonString(at.targetFunction));
        jsonPush(arr, std::move(entry));
    }
    return arr;
}

JsonValue threadResultToJson(const ThreadGroupResult &tg) {
    JsonValue entry = jsonObject();
    jsonPut(entry, "threads", jsonInt(tg.numThreads));
    jsonPut(entry, "status", jsonString(tg.status));
    jsonPut(entry, "source_runs", jsonInt(tg.originalRuns));
    jsonPut(entry, "transformed_runs", jsonInt(tg.transformedRuns));
    jsonPut(entry, "outcome_set", outcomeSetResultToJson(tg.outcomeSet));
    return entry;
}

// short category of the first non-OK thread, or the pipeline error itself
std::string runMessage(const RunInfo &info) {
    for (const auto &tg : info.threadResults)
        if (tg.status != "OK")
            return tg.message;
    if (!info.error.empty())
        return info.error;
    return info.warn;
}

JsonValue relationToJson(OutcomeRelation relation) {
    JsonValue obj = jsonObject();
    jsonPut(obj, "type", jsonString(outcomeRelationToString(relation)));
    return obj;
}

// metadata fields shared by every run-info JSON representation; field order is
// significant (ordered JsonValue), so per-representation fields must be
// appended after this
JsonValue runMetadataToJson(const RunInfo &info) {
    JsonValue obj = jsonObject();
    jsonPut(obj, "run", jsonInt(info.runNumber));
    jsonPut(obj, "file", jsonString(info.file));
    jsonPut(obj, "applied", appliedTransformsToJson(info.appliedTransforms));
    jsonPut(obj, "relation", relationToJson(info.relation));

    std::string status = runStatusString(info);
    jsonPut(obj, "status", jsonString(status));

    std::string message = runMessage(info);
    if (!message.empty())
        jsonPut(obj, "message", jsonString(message));

    jsonPut(obj, "seed", jsonInt(info.seed));
    jsonPut(obj, "requested_transforms", requestedTransformsToJson(info));
    return obj;
}

} // namespace

JsonValue runInfoToJson(const RunInfo &info) {
    return runMetadataToJson(info);
}

std::string runStatusString(const RunInfo &info) {
    if (!info.error.empty())
        return "ERROR";
    for (const auto &tg : info.threadResults)
        if (tg.status == "ERROR")
            return "ERROR";
    for (const auto &tg : info.threadResults)
        if (tg.status == "WARN")
            return "WARN";
    if (!info.warn.empty())
        return "WARN";
    return "OK";
}

JsonValue runInfoToStatusJson(const RunInfo &info) {
    JsonValue obj = runMetadataToJson(info);

    JsonValue threads = jsonArray();
    for (const auto &tg : info.threadResults)
        jsonPush(threads, threadResultToJson(tg));
    jsonPut(obj, "thread_results", std::move(threads));
    return obj;
}

JsonValue runInfoToUnionJson(const RunInfo &info) {
    JsonValue obj = runMetadataToJson(info);

    jsonPut(obj, "source_runs", jsonInt(info.sourceRuns));
    jsonPut(obj, "transformed_runs", jsonInt(info.transformedRuns));
    jsonPut(obj, "source_outcomes",
            outcomeListToJson(info.sourceOutcomes, info.sourceCounts));
    jsonPut(obj, "transformed_outcomes",
            outcomeListToJson(info.transformedOutcomes,
                              info.transformedCounts));

    JsonValue binaries = jsonArray();
    for (const auto &b : info.binaryOutcomes) {
        JsonValue entry = jsonObject();
        jsonPut(entry, "side", jsonString(b.side));
        jsonPut(entry, "compile_index", jsonInt(b.compileIndex));
        jsonPut(entry, "jit_opt_level", jsonInt(b.jitOptLevel));
        jsonPut(entry, "runs", jsonInt(b.runs));
        jsonPut(entry, "outcomes",
                outcomeListToJson(b.outcomes, b.counts));
        jsonPush(binaries, std::move(entry));
    }
    jsonPut(obj, "binary_results", std::move(binaries));
    return obj;
}

JsonValue executionRunToJson(const ExecutionRunResult &run) {
    JsonValue obj = jsonObject();
    jsonPut(obj, "run", jsonInt(run.runNumber));
    jsonPut(obj, "file", jsonString(run.file));
    jsonPut(obj, "seed", jsonInt(run.seed));
    JsonValue threadResults = jsonArray();
    for (const auto &tr : run.threadResults) {
        JsonValue entry = jsonObject();
        jsonPut(entry, "threads", jsonInt(tr.numThreads));
        jsonPut(entry, "runs", jsonInt(tr.runs));
        jsonPut(entry, "outcomes", outcomeListToJson(tr.outcomes, tr.counts));
        jsonPush(threadResults, std::move(entry));
    }
    jsonPut(obj, "thread_results", std::move(threadResults));
    if (!run.error.empty())
        jsonPut(obj, "error", jsonString(run.error));
    return obj;
}

} // namespace mlir_mracle

