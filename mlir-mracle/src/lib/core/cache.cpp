#include "mlir-mracle/core/cache.h"

#include "mlir-mracle/io/io.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace mlir_mracle {

bool cachingEnabled() {
    if (const char *dir = std::getenv("MLIR_MRACLE_CACHE_DIR"))
        return *dir != '\0';
    return true;
}

std::filesystem::path cacheRoot() {
    if (!cachingEnabled())
        return {};
    // the runner pins the root so every invocation shares the same cache
    // regardless of the working directory
    if (const char *dir = std::getenv("MLIR_MRACLE_CACHE_DIR"))
        return dir;
    return "cache/v2";
}

std::filesystem::path moduleCacheDir() {
    return cacheRoot() / "modules";
}

std::filesystem::path baselineCacheDir() {
    return cacheRoot() / "baselines";
}

std::filesystem::path moduleCachePath(const std::string &hash) {
    return moduleCacheDir() / (hash + ".bc");
}

std::filesystem::path loweredCachePath(const std::string &hash) {
    return moduleCacheDir() / (hash + ".lowered.mlir");
}

std::filesystem::path llvmIRCachePath(const std::string &hash) {
    return moduleCacheDir() / (hash + ".ll");
}

// the baseline cache key includes the sampling budget so a campaign run with
// a different --reps never reuses a baseline collected with another budget.
// The key also embeds the result schema version both in the hash input and in
// the file-name prefix, so a format change neither reuses nor orphans stale
// entries: the prefix lets ensureCacheDirs sweep them before a new campaign
// starts.
std::filesystem::path baselineCachePath(const std::string &hash,
                                        const std::string &variant, int reps) {
    std::string versionedHash =
        hashString(hash + "|schema:" + std::to_string(kResultSchemaVersion));
    return baselineCacheDir() /
           ("v" + std::to_string(kResultSchemaVersion) + "_" + versionedHash +
            "_r" + std::to_string(reps) + "_" + variant + ".json");
}

// removes baseline cache files left by an older result schema: the current
// schema prefixes its baseline files with "v<N>_", so any other .json in the
// baseline cache is stale. Runs once per process, before the first baseline
// lookup or write.
void sweepStaleBaselines() {
    if (!cachingEnabled())
        return;
    static bool swept = false;
    if (swept)
        return;
    swept = true;
    const std::string prefix =
        "v" + std::to_string(kResultSchemaVersion) + "_";
    std::error_code ec;
    for (const auto &entry :
         std::filesystem::directory_iterator(baselineCacheDir(), ec)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".json" &&
            entry.path().filename().string().rfind(prefix, 0) != 0)
            std::filesystem::remove(entry.path(), ec);
    }
}

void ensureCacheDirs() {
    if (!cachingEnabled())
        return;
    std::error_code ec;
    std::filesystem::create_directories(moduleCacheDir(), ec);
    std::filesystem::create_directories(baselineCacheDir(), ec);
    sweepStaleBaselines();
}

bool readFileContent(const std::string &path, std::string &content) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    content.assign(std::istreambuf_iterator<char>(f),
                   std::istreambuf_iterator<char>());
    return true;
}

bool writeFileContent(const std::string &path, const std::string &content) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        return false;
    f << content;
    return true;
}

std::string hashString(const std::string &s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    std::string hex;
    for (int shift = 60; shift >= 0; shift -= 4)
        hex += "0123456789abcdef"[(h >> shift) & 0xF];
    return hex;
}

bool loadCachedModule(const std::string &path, llvm::LLVMContext &ctx,
                      std::unique_ptr<llvm::Module> &module,
                      std::string &error) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(path);
    if (!bufOrErr)
        return false;
    auto modOrErr = llvm::parseBitcodeFile((*bufOrErr)->getMemBufferRef(), ctx);
    if (!modOrErr) {
        error = llvm::toString(modOrErr.takeError());
        return false;
    }
    module = std::move(*modOrErr);
    return true;
}

void saveCachedModule(const std::string &path, llvm::Module &module) {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec)
        return;
    llvm::WriteBitcodeToFile(module, os);
}

// loads a cached module and its artifact sidecars for the given hash; returns
// false when no cache entry exists
bool loadModuleCache(const std::string &hash, llvm::LLVMContext &ctx,
                     std::unique_ptr<llvm::Module> &module,
                     std::string *loweredMLIR, std::string *llvmIR,
                     std::string *bitcode, std::string &error) {
    if (!cachingEnabled())
        return false;
    if (!loadCachedModule(moduleCachePath(hash).string(), ctx, module, error))
        return false;
    if (loweredMLIR)
        readFileContent(loweredCachePath(hash).string(), *loweredMLIR);
    if (llvmIR)
        readFileContent(llvmIRCachePath(hash).string(), *llvmIR);
    if (bitcode)
        readFileContent(moduleCachePath(hash).string(), *bitcode);
    return true;
}

void saveModuleCache(const std::string &hash, llvm::Module &module,
                     const std::string &loweredMLIR,
                     const std::string &llvmIR) {
    if (!cachingEnabled())
        return;
    ensureCacheDirs();
    saveCachedModule(moduleCachePath(hash).string(), module);
    writeFileContent(loweredCachePath(hash).string(), loweredMLIR);
    writeFileContent(llvmIRCachePath(hash).string(), llvmIR);
}

bool loadBaselineCache(const std::string &sourceHash,
                       const std::string &variant, int reps,
                       ObservedOutcomeSet &set) {
    if (!cachingEnabled())
        return false;
    std::string content;
    if (!readFileContent(baselineCachePath(sourceHash, variant, reps).string(),
                         content))
        return false;
    auto parsed = llvm::json::parse(content);
    if (!parsed)
        return false;
    const auto *obj = parsed->getAsObject();
    if (!obj)
        return false;
    return observedOutcomeSetFromJson(*obj, set);
}

void saveBaselineCache(const std::string &sourceHash,
                       const std::string &variant, int reps,
                       const ObservedOutcomeSet &set) {
    if (!cachingEnabled())
        return;
    ensureCacheDirs();
    std::string buf;
    llvm::raw_string_ostream os(buf);
    printJson(observedOutcomeSetToJson(set), os);
    os << "\n";
    os.flush();
    writeFileContent(baselineCachePath(sourceHash, variant, reps).string(),
                     buf);
}

} // namespace mlir_mracle
