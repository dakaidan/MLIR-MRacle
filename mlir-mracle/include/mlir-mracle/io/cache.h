#pragma once

#include "mlir-mracle/context/context.h"

#include "llvm/IR/Module.h"

#include <filesystem>
#include <memory>
#include <string>

namespace mlir_mracle {

// schema version embedded in persistent baseline-cache keys and prefixes
inline constexpr int kResultSchemaVersion = 8;

// stable 64-bit FNV-1a hash rendered as hex; cache keys only need to match
// within a machine, not be cryptographic
std::string hashString(const std::string &s);

// An explicitly empty MLIR_MRACLE_CACHE_DIR disables the disk cache entirely;
// an unset variable uses the default cache/v2 root.
bool cachingEnabled();

std::filesystem::path cacheRoot();
std::filesystem::path moduleCacheDir();
std::filesystem::path baselineCacheDir();
std::filesystem::path moduleCachePath(const std::string &hash);
std::filesystem::path loweredCachePath(const std::string &hash);
std::filesystem::path llvmIRCachePath(const std::string &hash);
std::filesystem::path baselineCachePath(const std::string &hash,
                                        const std::string &variant, int reps);

void sweepStaleBaselines();
void ensureCacheDirs();

bool readFileContent(const std::string &path, std::string &content);
bool writeFileContent(const std::string &path, const std::string &content);

bool loadCachedModule(const std::string &path, llvm::LLVMContext &ctx,
                      std::unique_ptr<llvm::Module> &module,
                      std::string &error);
void saveCachedModule(const std::string &path, llvm::Module &module);

bool loadModuleCache(const std::string &hash, llvm::LLVMContext &ctx,
                     std::unique_ptr<llvm::Module> &module,
                     std::string *loweredMLIR, std::string *llvmIR,
                     std::string *bitcode, std::string &error);
void saveModuleCache(const std::string &hash, llvm::Module &module,
                     const std::string &loweredMLIR,
                     const std::string &llvmIR);

bool loadBaselineCache(const std::string &sourceHash,
                       const std::string &variant, int reps,
                       ObservedOutcomeSet &set);
void saveBaselineCache(const std::string &sourceHash,
                       const std::string &variant, int reps,
                       const ObservedOutcomeSet &set);

} // namespace mlir_mracle
