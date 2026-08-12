#include "mlir-mr/io/io.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir_mr {

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

std::string formatRunInfo(const RunInfo &info) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "run: " << info.runNumber << "\n";
    os << "seed: " << info.seed << "\n";
    os << "file: " << info.file << "\n";
    os << "requested-transforms: ";
    if (info.requestedTransforms.empty())
        os << "all\n";
    else
        os << llvm::join(info.requestedTransforms, ",") << "\n";
    if (info.transformApplied)
        for (const auto &at : info.appliedTransforms)
            os << "applied-transformation: " << at.name
               << " in function '" << at.targetFunction << "'\n";
    else
        os << "applied-transformation: none\n";
    for (const auto &tg : info.threadResults)
        os << "threads: " << tg.numThreads
           << " ok: " << (tg.comparison.ok ? "yes" : "no")
           << " warn: " << (tg.comparison.warn ? "yes" : "no")
           << " " << tg.comparison.message << "\n";
    if (!info.error.empty())
        os << "error: " << info.error << "\n";
    if (!info.warn.empty())
        os << "warn: " << info.warn << "\n";
    return buf;
}

llvm::json::Object fisherResultToJson(const FisherResult &fr) {
    llvm::json::Object vobj;
    vobj["p_value"] = fr.pValue;
    vobj["p_low"] = fr.pLow;
    vobj["p_high"] = fr.pHigh;
    vobj["simulations"] = static_cast<int64_t>(fr.simulations);
    llvm::json::Array cats, srcC, trC, novel, dis;
    for (size_t i = 0; i < fr.categories.size(); ++i) {
        cats.push_back(llvm::json::Value(fr.categories[i]));
        srcC.push_back(llvm::json::Value(static_cast<int64_t>(fr.sourceCounts[i])));
        trC.push_back(llvm::json::Value(static_cast<int64_t>(fr.transformedCounts[i])));
    }
    for (auto v : fr.novelOutcomes)
        novel.push_back(llvm::json::Value(v));
    for (auto v : fr.disappearedOutcomes)
        dis.push_back(llvm::json::Value(v));
    vobj["categories"] = std::move(cats);
    vobj["source_counts"] = std::move(srcC);
    vobj["transformed_counts"] = std::move(trC);
    vobj["novel_outcomes"] = std::move(novel);
    vobj["disappeared_outcomes"] = std::move(dis);
    return vobj;
}

bool fisherResultFromJson(const llvm::json::Object &o, FisherResult &fr) {
    auto p = o.getNumber("p_value");
    if (!p)
        return false;
    fr.pValue = *p;
    if (auto pl = o.getNumber("p_low"))
        fr.pLow = *pl;
    if (auto ph = o.getNumber("p_high"))
        fr.pHigh = *ph;
    if (auto s = o.getInteger("simulations"))
        fr.simulations = static_cast<int>(*s);
    if (const auto *cats = o.getArray("categories"))
        for (const auto &cv : *cats)
            if (auto v = cv.getAsInteger())
                fr.categories.push_back(static_cast<int64_t>(*v));
    if (const auto *sc = o.getArray("source_counts"))
        for (const auto &cv : *sc)
            if (auto v = cv.getAsInteger())
                fr.sourceCounts.push_back(static_cast<int>(*v));
    if (const auto *tc = o.getArray("transformed_counts"))
        for (const auto &cv : *tc)
            if (auto v = cv.getAsInteger())
                fr.transformedCounts.push_back(static_cast<int>(*v));
    if (const auto *nov = o.getArray("novel_outcomes"))
        for (const auto &cv : *nov)
            if (auto v = cv.getAsInteger())
                fr.novelOutcomes.push_back(static_cast<int64_t>(*v));
    if (const auto *dis = o.getArray("disappeared_outcomes"))
        for (const auto &cv : *dis)
            if (auto v = cv.getAsInteger())
                fr.disappearedOutcomes.push_back(static_cast<int64_t>(*v));
    return true;
}

llvm::json::Object runInfoToJson(const RunInfo &info, bool includeMLIR) {
    llvm::json::Object obj;
    obj["run"] = info.runNumber;
    obj["seed"] = info.seed;
    obj["file"] = info.file;
    llvm::json::Array requested;
    if (info.requestedTransforms.empty())
        requested.push_back("all");
    else
        for (const auto &r : info.requestedTransforms)
            requested.push_back(r);
    obj["requested_transforms"] = std::move(requested);
    llvm::json::Array applied;
    for (const auto &at : info.appliedTransforms) {
        llvm::json::Object entry;
        entry["name"] = at.name;
        entry["target_function"] = at.targetFunction;
        applied.push_back(std::move(entry));
    }
    obj["applied_transforms"] = std::move(applied);
    obj["transform_applied"] = info.transformApplied;
    llvm::json::Array threadResults;
    int originalRuns = 0, transformedRuns = 0;
    for (const auto &tg : info.threadResults) {
        llvm::json::Object entry;
        entry["threads"] = tg.numThreads;
        entry["ok"] = tg.comparison.ok;
        entry["warn"] = tg.comparison.warn;
        entry["message"] = tg.comparison.message;
        entry["original_runs"] = tg.originalRuns;
        entry["transformed_runs"] = tg.transformedRuns;
        if (!tg.variables.empty()) {
            double worstP = 1.0;
            llvm::json::Array vars;
            for (const auto &fr : tg.variables) {
                vars.push_back(fisherResultToJson(fr));
                worstP = std::min(worstP, fr.pValue);
            }
            entry["p_value"] = worstP;
            entry["variables"] = std::move(vars);
        }
        threadResults.push_back(std::move(entry));
        originalRuns = tg.originalRuns;
        transformedRuns = tg.transformedRuns;
    }
    obj["thread_results"] = std::move(threadResults);
    obj["original_runs"] = originalRuns;
    obj["transformed_runs"] = transformedRuns;
    if (!info.error.empty())
        obj["error"] = info.error;
    if (!info.warn.empty())
        obj["warn"] = info.warn;
    if (includeMLIR)
        obj["mlir_output"] = info.mlirOutput;
    return obj;
}

} // namespace mlir_mr
