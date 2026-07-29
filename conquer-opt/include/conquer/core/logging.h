#pragma once

#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <atomic>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

#include "mlir/IR/OpDefinition.h"

namespace conquer {

// Runtime logging level used by L_INFO/L_DEBUG/L_TRACE.
// -1: disabled, 0: info, 1: debug, 2: trace
inline std::atomic<int> &log_level() {
    static std::atomic<int> level{-1};
    return level;
}

inline void set_log_level(const int level) {
    // Supported levels: -1 (off), 0 (info), 1 (debug), 2 (trace), 4 (enable LLVM/MLIR/IREE debug flag in CLI).
    int clamped = level;
    if (clamped < -1) clamped = -1;
    if (clamped > 4) clamped = 4;
    log_level().store(clamped, std::memory_order_relaxed);
}

inline int get_log_level() {
    return log_level().load(std::memory_order_relaxed);
}

inline std::vector<std::string> &debug_only_filters() {
    static std::vector<std::string> filters;
    return filters;
}

inline std::string trim_copy(const std::string &s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }
    return s.substr(start, end - start);
}

inline void set_debug_only_filters(const std::string &debug_only_csv) {
    auto &filters = debug_only_filters();
    filters.clear();

    std::stringstream ss(debug_only_csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (!item.empty()) {
            filters.push_back(item);
        }
    }
}

inline bool matches_debug_only_filter(const std::string_view debug_type) {
    const auto &filters = debug_only_filters();
    if (filters.empty()) {
        return true;
    }

    for (const auto &f : filters) {
        if (debug_type == f) {
            return true;
        }
    }
    return false;
}

inline bool should_log(const int required_level, const std::string_view debug_type) {
    const int level = get_log_level();
    if (level < required_level) {
        return false;
    }
    return matches_debug_only_filter(debug_type);
}

// Level-aware logging macros.
// They are controlled by ConQuER runtime settings (set_log_level + set_debug_only_filters).
// - L_INFO:  Informational messages (e.g. starting a pass, loading a model).
// - L_DEBUG: Detailed debugging info (e.g. op-level decisions, repairs).
// - L_TRACE: Highly verbose output (e.g. per-iteration data, full IR dumps).
//
// Usage:
//   #undef DEBUG_TYPE
//   #define DEBUG_TYPE "my-component"
//   ...
//   L_INFO("Something happened: " << some_value);
//
// Filtering:
//   --debug-level controls verbosity for these macros (0..2).
//   --debug-only filters by exact base DEBUG_TYPE names (comma separated).
//   --debug-level 4 additionally enables raw LLVM/MLIR/IREE LLVM_DEBUG output.

#define L_INFO(msg) \
  do { \
    if (conquer::should_log(0, DEBUG_TYPE)) { \
      llvm::dbgs() << "[" DEBUG_TYPE "][INFO] " << msg << "\n"; \
    } \
  } while (0)

#define L_DEBUG(msg) \
  do { \
    if (conquer::should_log(1, DEBUG_TYPE)) { \
      llvm::dbgs() << "[" DEBUG_TYPE "][DEBUG] " << msg << "\n"; \
    } \
  } while (0)

#define L_TRACE(msg) \
  do { \
    if (conquer::should_log(2, DEBUG_TYPE)) { \
      llvm::dbgs() << "[" DEBUG_TYPE "][TRACE] " << msg << "\n"; \
    } \
  } while (0)

// Helper to print MLIR operations compactly to avoid clogging the logs.
inline std::string compact(mlir::Operation *op) {
    if (!op) return "null";
    std::string s;
    llvm::raw_string_ostream os(s);
    os << op->getName();

    // Print types (results)
    if (op->getNumResults() > 0) {
        os << " : (";
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
            os << op->getResult(i).getType();
            if (i < op->getNumResults() - 1) os << ", ";
        }
        os << ")";
    }

    return os.str();
}

inline std::string compact(const mlir::Value val) {
    if (!val) return "null";
    std::string s;
    llvm::raw_string_ostream os(s);
    os << val.getType();
    if (const auto op = val.getDefiningOp()) {
        os << " (from " << compact(op) << ")";
    } else {
        os << " (arg)";
    }
    return os.str();
}

template <typename T>
requires std::derived_from<T, mlir::OpState>
inline std::string compact(T op) {
    return compact(op.getOperation());
}

} // namespace conquer
