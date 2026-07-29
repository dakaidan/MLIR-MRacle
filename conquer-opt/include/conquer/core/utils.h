#pragma once
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MD5.h>

#include <fstream>
#include <sstream>
#include <string>

namespace conquer {
inline std::string load_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

inline std::string get_cached(const std::string &input, const std::string &tag = std::string()) {
    llvm::MD5 hash;
    hash.update(input);

    llvm::MD5::MD5Result hashResult;
    hash.final(hashResult);
    llvm::SmallString<32> hashHex;
    llvm::MD5::stringifyResult(hashResult, hashHex);

    if (const std::string cachePath = "/tmp/conquer_" + tag + "_" + std::string(hashHex.c_str());
        llvm::sys::fs::exists(cachePath)) {
        return load_file(cachePath);
    }
    return "";
}

inline bool is_cached(const std::string &input, const std::string &tag = std::string()) {
    llvm::MD5 hash;
    hash.update(input);

    llvm::MD5::MD5Result hashResult;
    hash.final(hashResult);
    llvm::SmallString<32> hashHex;
    llvm::MD5::stringifyResult(hashResult, hashHex);

    const std::string cachePath = "/tmp/conquer_" + tag + "_" + std::string(hashHex.c_str());

    return llvm::sys::fs::exists(cachePath);
}

struct cache_result {
    bool is_cached;
    std::string result;
};

template <typename Func>
cache_result cache_result(const std::string &key, Func func, const std::string &tag = std::string()) {
    if (is_cached(key, tag)) {
        return {true, get_cached(key, tag)};
    }
    llvm::MD5 hash;
    hash.update(key);

    llvm::MD5::MD5Result hashResult;
    hash.final(hashResult);
    llvm::SmallString<32> hashHex;
    llvm::MD5::stringifyResult(hashResult, hashHex);

    const std::string result = func();
    std::error_code ec;
    llvm::raw_fd_ostream os("/tmp/conquer_" + tag + "_" + std::string(hashHex.c_str()), ec, llvm::sys::fs::OF_None);
    if (!ec) {
        os << result;
    } else {
        throw std::runtime_error("Could not write cache file for key: " + std::string(hashHex.c_str()));
    }
    return {false, result};
}
} // namespace conquer