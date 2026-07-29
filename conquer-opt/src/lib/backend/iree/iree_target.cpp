#include "conquer/backend/iree/iree_target.h"
#include "conquer/core/logging.h"

#include "conquer/backend/iree/iree_wrapper.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <iree/compiler/mlir_interop.h>
#pragma clang diagnostic pop

#include <fstream>
#include <mlir/Bytecode/BytecodeWriter.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-iree-target"

conquer::IREETarget::IREETarget(const IREETargetBackend backend) {
    L_DEBUG("Creating IREE target for backend: " << to_string(backend));
    ensure_iree_initialized();
    this->backend_ = backend;
    switch (backend_) {
    case IREETargetBackend::CPU:
        L_DEBUG("Querying CPU capabilities...");
        this->target_capability = query_cpu_capability();
        break;
    case IREETargetBackend::CUDA:
        L_DEBUG("Querying CUDA capabilities...");
        this->target_capability = query_cuda_capability();
        break;
    case IREETargetBackend::ROCM:
        L_DEBUG("Querying ROCm capabilities...");
        this->target_capability = query_rocm_capability();
        break;
    case IREETargetBackend::VULKAN:
        L_DEBUG("Querying Vulkan capabilities...");
        this->target_capability = query_vulkan_capability();
        break;
    default:
        L_DEBUG("Unable to query adb fallback to CPI capabilities");
        this->target_capability = query_cpu_capability();
        break;
    }
}

conquer::IREETarget::IREETarget(IREETarget &&other) noexcept {
    L_DEBUG("Moving IREE target for backend: " << to_string(other.backend_));
    backend_ = other.backend_;
    compiled_module_data_ = std::move(other.compiled_module_data_);
    target_capability = other.target_capability;
    runtime_session_ = std::move(other.runtime_session_);

    other.backend_ = IREETargetBackend::CPU; // Default to CPU, but it doesn't really matter as the moved-from object
                                             // shouldn't be used.
    other.compiled_module_data_.clear();
    other.target_capability = HardwareCapability{};
}

conquer::IREETarget &conquer::IREETarget::operator=(IREETarget &&other) noexcept {
    if (this != &other) {
        backend_ = other.backend_;
        compiled_module_data_ = std::move(other.compiled_module_data_);
        target_capability = other.target_capability;
        runtime_session_ = std::move(other.runtime_session_);

        other.backend_ = IREETargetBackend::CPU; // Default to CPU, but it doesn't really matter as the moved-from
                                                 // object shouldn't be used.
        other.compiled_module_data_.clear();
        other.target_capability = HardwareCapability{};
    }
    return *this;
}

std::string derive_vulkan_target(const std::string &device_name) {
    std::string name = device_name;
    std::ranges::transform(name, name.begin(), [](const unsigned char c) { return std::tolower(c); });

    std::smatch match;
    if (std::regex_search(name, match, std::regex(R"(gfx\d{4})"))) {
        return match.str(0);
    }

    if (std::regex_search(name, match, std::regex(R"((rtx|gtx)[\s-]?(\d{4}))"))) {
        return match[1].str() + match[2].str();
    }

    return "rdna2";
}

llvm::Error conquer::IREETarget::compile(llvm::StringRef bytecode) {
    L_INFO("Starting compilation for IREE target (" << to_string(backend_) << ").");
    // Reset any active session to drop hardware bindings for the old module
    runtime_session_.reset();

    IREECompilerSession compiler_session;

    const auto capabilities = this->query_capability();

    // Incredibly unsure if this is actually applying the peel or not
    std::vector<const char*> global_flags = {
        "iree-compiler",
        "--iree-llvmcpu-vector-pproc-strategy=peel",
        "--iree-llvmcpu-enable-scalable-vectorization"
    };

    ireeCompilerSetupGlobalCL(
        static_cast<int>(global_flags.size()),
        global_flags.data(),
        "IREE Compiler Global Flags",
        false
    );

    std::vector<std::string> flags;
    flags.push_back("--iree-hal-target-backends=" + std::string(to_string(backend_)));
    flags.push_back("--iree-input-type=tosa");

    if (backend_ == IREETargetBackend::CPU) {
        L_DEBUG("Adding host CPU target flags.");
        flags.emplace_back("--iree-llvmcpu-target-cpu=host");
        flags.emplace_back("--iree-opt-data-tiling=true");
    }
    else if (backend_ == IREETargetBackend::ADB_CPU) {
        L_DEBUG("Adding Snapdragon CPU cross-compile flags (SME enabled).");
        flags.emplace_back("--iree-llvmcpu-target-triple=aarch64-none-linux-android");
        flags.emplace_back("--iree-opt-data-tiling=false");
        flags.emplace_back("--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-preprocessing-transpose-matmul-pass{input=lhs}))");
    }
    else if (backend_ == IREETargetBackend::VULKAN) {
        const std::string target_string = derive_vulkan_target(capabilities.device_name);
        L_DEBUG("Derived IREE Vulkan Target '" << target_string << "'");
        flags.push_back("--iree-vulkan-target=" + target_string);
    }
    else if (backend_ == IREETargetBackend::ADB_GPU) {
        L_DEBUG("Adding Snapdragon Adreno GPU flags.");
        flags.push_back("--iree-vulkan-target=adreno");
        flags.push_back("--iree-stream-resource-max-allocation-size=1073741824"); // 1GB allocation limit
    }
    else if (backend_ == IREETargetBackend::CUDA) {
        flags.push_back("--iree-cuda-target=sm_80"); // sm_80 is A100
    }

    if (auto err = compiler_session.set_flags(flags)) {
        return err;
    }

    if (auto err = compiler_session.wrap_module_bytecode(bytecode)) {
        return err;
    }
    if (!ireeCompilerInvocationPipeline(compiler_session.invocation(), IREE_COMPILER_PIPELINE_STD)) {
        return llvm::make_error<llvm::StringError>("IREE standard compilation pipeline failed. "
                                                   "Check console output for MLIR diagnostic errors.",
                                                   llvm::inconvertibleErrorCode());
    }

    if (auto err = compiler_session.open_output_membuffer()) {
        return err;
    }

    iree_compiler_error_t *c_err =
        ireeCompilerInvocationOutputVMBytecode(compiler_session.invocation(), compiler_session.output());
    if (c_err) {
        return make_llvm_error(c_err);
    }

    auto expected_bytes = compiler_session.extract_output();
    if (!expected_bytes) {
        return expected_bytes.takeError();
    }

    L_INFO("Successfully compiled module for IREE target (" << to_string(backend_)
                            << "). Compiled module size: " << expected_bytes->size() << " bytes.");

    compiled_module_data_ = std::move(expected_bytes.get());

    return llvm::Error::success();
}

llvm::Error conquer::IREETarget::compile(mlir::Operation *module) {
    LLVM_DEBUG(llvm::dbgs() << "Starting compilation for IREE target (" << to_string(backend_) << ").\n");

    // Reset any active session to drop hardware bindings for the old module
    runtime_session_.reset();

    IREECompilerSession compiler_session;

    const auto capabilities = this->query_capability();

    // Incredibly unsure if this is actually applying the peel or not
    std::vector<const char*> global_flags = {
        "iree-compiler",
        "--iree-llvmcpu-vector-pproc-strategy=peel",
        "--iree-llvmcpu-enable-scalable-vectorization"
    };

    ireeCompilerSetupGlobalCL(
        static_cast<int>(global_flags.size()),
        global_flags.data(),
        "IREE Compiler Global Flags",
        false
    );

    std::vector<std::string> flags;
    flags.push_back("--iree-hal-target-backends=" + std::string(to_string(backend_)));
    flags.push_back("--iree-input-type=tosa");

    if (backend_ == IREETargetBackend::CPU) {
        L_DEBUG("Adding host CPU target flags.");
        flags.emplace_back("--iree-llvmcpu-target-cpu=host");
        flags.emplace_back("--iree-opt-data-tiling=true");
    }
    else if (backend_ == IREETargetBackend::ADB_CPU) {
        L_DEBUG("Adding Snapdragon CPU cross-compile flags (SME enabled).");
        flags.emplace_back("--iree-llvmcpu-target-triple=aarch64-none-linux-android");
        flags.emplace_back("--iree-opt-data-tiling=false");
        flags.emplace_back("--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-preprocessing-transpose-matmul-pass{input=lhs}))");
    }
    else if (backend_ == IREETargetBackend::VULKAN) {
        const std::string target_string = derive_vulkan_target(capabilities.device_name);
        L_DEBUG("Derived IREE Vulkan Target '" << target_string << "'");
        flags.push_back("--iree-vulkan-target=" + target_string);
    }
    else if (backend_ == IREETargetBackend::ADB_GPU) {
        L_DEBUG("Adding Snapdragon Adreno GPU flags.");
        flags.push_back("--iree-vulkan-target=adreno");
        flags.push_back("--iree-stream-resource-max-allocation-size=1073741824"); // 1GB allocation limit
    }
    else if (backend_ == IREETargetBackend::CUDA) {
        flags.push_back("--iree-cuda-target=sm_80"); // sm_80 is A100
    }

    if (auto err = compiler_session.set_flags(flags)) {
        return err;
    }

    // TODO: Change this to a borrow module, but will need to change the initial module load path to use
    //       an iree context

    // WARNING: Masks errors when the module is invalid, but borrow is not possible without the dialects loaded
    // if (auto err = compiler_session.wrap_module_bytecode("tosa_module.mlirbc", module)) {
    //     return err;
    // }
    std::string code;
    {
        llvm::raw_string_ostream os(code);
        module->print(os);
    }

    if (auto err = compiler_session.wrap_source_buffer("tosa_module.mlir", code)) {
        return err;
    }

    if (!ireeCompilerInvocationPipeline(compiler_session.invocation(), IREE_COMPILER_PIPELINE_STD)) {
        return llvm::make_error<llvm::StringError>("IREE standard compilation pipeline failed. "
                                                   "Check console output for MLIR diagnostic errors.",
                                                   llvm::inconvertibleErrorCode());
    }

    if (auto err = compiler_session.open_output_membuffer()) {
        return err;
    }

    iree_compiler_error_t *c_err =
        ireeCompilerInvocationOutputVMBytecode(compiler_session.invocation(), compiler_session.output());
    if (c_err) {
        return make_llvm_error(c_err);
    }

    auto expected_bytes = compiler_session.extract_output();
    if (!expected_bytes) {
        return expected_bytes.takeError();
    }

    L_INFO("Successfully compiled module for IREE target (" << to_string(backend_)
                            << "). Compiled module size: " << expected_bytes->size() << " bytes.");

    compiled_module_data_ = std::move(expected_bytes.get());

    return llvm::Error::success();
}

llvm::Error conquer::IREETarget::save_compiled_module_to_file(const std::string &filename) const {
    if (compiled_module_data_.empty()) {
        L_DEBUG("No compiled module data to save. Did you call compile()?");
        return llvm::make_error<llvm::StringError>(
            "No compiled module data to save. Did you call compile()?",
            llvm::inconvertibleErrorCode());
    }

    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        L_DEBUG("Failed to open file for writing: " << filename);
        return llvm::make_error<llvm::StringError>(
            "Failed to open file for writing: " + filename,
            llvm::inconvertibleErrorCode());
    }

    outFile.write(reinterpret_cast<const char *>(compiled_module_data_.data()), compiled_module_data_.size());
    outFile.close();

    L_INFO("Compiled module saved to: " << filename);
    return llvm::Error::success();
}

llvm::Error conquer::IREETarget::load_compiled_module_from_file(const std::string &filename) {
    std::ifstream inFile(filename, std::ios::binary | std::ios::ate);
    if (!inFile) {
        L_DEBUG("Failed to open file for reading: " << filename);
        return llvm::make_error<llvm::StringError>(
            "Failed to open file for reading: " + filename,
            llvm::inconvertibleErrorCode());
    }

    const auto fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    if (!inFile) {
        L_DEBUG("Failed to seek in file: " << filename);
        return llvm::make_error<llvm::StringError>(
            "Failed to seek in file: " + filename,
            llvm::inconvertibleErrorCode());
    }

    compiled_module_data_.resize(fileSize);
    inFile.read(reinterpret_cast<char *>(compiled_module_data_.data()), fileSize);
    inFile.close();

    return llvm::Error::success();
}

llvm::Error conquer::IREETarget::execute(const std::vector<TensorView> &inputs,
                                         const std::vector<TensorView> &outputs) {
    L_INFO("Starting execution for IREE target (" << to_string(backend_) << ").");
    L_DEBUG("Number of inputs: " << inputs.size() << ", Number of outputs: " << outputs.size());

    if (compiled_module_data_.empty()) {
        return llvm::make_error<llvm::StringError>(
            "Execution failed: No compiled module data available. Did you call compile()?",
            llvm::inconvertibleErrorCode());
    }

    try {
        if (!runtime_session_) {
            LLVM_DEBUG(llvm::dbgs() << "Instantiating persistent IREERuntimeSession.\n");
            runtime_session_ = std::make_unique<IREERuntimeSession>(compiled_module_data_, backend_, target_capability.device_uri);
        }

        runtime_session_->invoke(inputs, outputs);

    } catch (const std::exception& e) {
        return llvm::make_error<llvm::StringError>(
            std::string("Execution threw an exception: ") + e.what(),
            llvm::inconvertibleErrorCode());
    }

    return llvm::Error::success();
}

conquer::HardwareCapability conquer::IREETarget::query_capability() const { return target_capability; }