#pragma once

#include "conquer/runtime/session.h"

#include <string>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <iree/compiler/embedding_api.h>
#include <iree/runtime/api.h>
#pragma clang diagnostic pop

#include <llvm/Support/Error.h>

namespace conquer {
enum class IREETargetBackend {
    VULKAN,
    CPU,
    CUDA,
    ROCM,
    ADB_CPU,
    ADB_GPU
    // Add more backends as needed
};

constexpr std::string_view to_string(const IREETargetBackend backend) {
    switch (backend) {
    case IREETargetBackend::VULKAN:
        return "vulkan-spirv";
    case IREETargetBackend::CPU:
        return "llvm-cpu";
    case IREETargetBackend::CUDA:
        return "cuda";
    case IREETargetBackend::ROCM:
        return "rocm";
    case IREETargetBackend::ADB_CPU:
        return "llvm-cpu";
    case IREETargetBackend::ADB_GPU:
        return "vulkan-spirv";
    }
    return "unknown";
}

struct IREEGlobalContext {
    IREEGlobalContext();
    ~IREEGlobalContext();
};

struct IREERuntimeInstance {
    iree_runtime_instance_t *instance = nullptr;

    IREERuntimeInstance();
    ~IREERuntimeInstance();
};

inline IREERuntimeInstance &get_iree_instance() {
    static IREERuntimeInstance instance;
    return instance;
}

// Initialises the global iree context
inline void ensure_iree_initialized() { static IREEGlobalContext global_context; }

class IREECompilerSession {
  public:
    IREECompilerSession();
    ~IREECompilerSession();

    IREECompilerSession(const IREECompilerSession &) = delete;
    IREECompilerSession &operator=(const IREECompilerSession &) = delete;

    IREECompilerSession(IREECompilerSession &&other) noexcept;
    IREECompilerSession &operator=(IREECompilerSession &&other) noexcept;

    iree_compiler_session_t *session() const { return session_; }
    iree_compiler_invocation_t *invocation() const { return inv_; }
    iree_compiler_source_t *source() const { return source_; }
    iree_compiler_output_t *output() const { return output_; }

    // Pipeline execution helpers
    llvm::Error wrap_source_buffer(const std::string &name, const std::string &buffer);
    llvm::Error wrap_module_bytecode(llvm::StringRef bytecode);
    llvm::Error wrap_module_bytecode(const std::string &name, mlir::Operation *module);
    llvm::Error import_steal_module(mlir::OwningOpRef<mlir::ModuleOp> module);
    llvm::Error import_borrow_module(mlir::Operation *module);
    llvm::Error open_output_membuffer();

    llvm::Error set_flags(const std::vector<std::string> &flags) const;
    llvm::Expected<std::vector<uint8_t>> extract_output() const;

  private:
    iree_compiler_session_t *session_ = nullptr;
    iree_compiler_invocation_t *inv_ = nullptr;
    iree_compiler_source_t *source_ = nullptr;
    iree_compiler_output_t *output_ = nullptr;
};

llvm::Error make_llvm_error(iree_compiler_error_t *iree_error);

iree_hal_element_type_t mapConquerTypeToIREE(DataType type);

class IREERuntimeSession {
  public:
    IREERuntimeSession(const std::vector<uint8_t> &module, IREETargetBackend backend, const std::string &device_uri);
    ~IREERuntimeSession();

    IREERuntimeSession(const IREERuntimeSession &) = delete;
    IREERuntimeSession &operator=(const IREERuntimeSession &) = delete;

    IREERuntimeSession(IREERuntimeSession &&other) noexcept;
    IREERuntimeSession &operator=(IREERuntimeSession &&other) noexcept;

    void invoke(const std::vector<TensorView> &inputs, const std::vector<TensorView> &outputs);

  private:
    iree_runtime_session_t *session_ = nullptr;
    iree_runtime_call_t call;

    void prepare_inputs(const std::vector<TensorView> &inputs);
    void prepare_outputs(const std::vector<TensorView> &outputs);
};
} // namespace conquer