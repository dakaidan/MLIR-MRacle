#include "conquer/backend/iree/iree_wrapper.h"
#include "conquer/core/logging.h"

#include <mlir/Bytecode/BytecodeWriter.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

#include <stdexcept>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <iree/compiler/mlir_interop.h>
#pragma clang diagnostic pop

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-iree"

conquer::IREEGlobalContext::IREEGlobalContext() {
#if IREE_ENABLED
    L_DEBUG("Initializing global IREE compiler context.");
    ireeCompilerGlobalInitialize();
#endif
}

conquer::IREEGlobalContext::~IREEGlobalContext() {
#if IREE_ENABLED
    L_DEBUG("Shutting down global IREE compiler context.");
    ireeCompilerGlobalShutdown();
#endif
}

conquer::IREERuntimeInstance::IREERuntimeInstance() {
#if IREE_ENABLED
    L_DEBUG("Initializing global IREE runtime instance.");
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);
    instance = nullptr;
    IREE_CHECK_OK(iree_runtime_instance_create(&instance_options, iree_allocator_system(), &instance));
#endif
}

conquer::IREERuntimeInstance::~IREERuntimeInstance() {
#if IREE_ENABLED
    L_DEBUG("Releasing global IREE runtime instance.");
    if (instance) {
        iree_runtime_instance_release(instance);
        instance = nullptr;
    }
#endif
}

conquer::IREECompilerSession::IREECompilerSession() {
    L_DEBUG("Creating IREE compiler session and invocation.");
    session_ = ireeCompilerSessionCreate();
    if (!session_) {
        throw std::runtime_error("Failed to create IREE compiler session");
    }
    inv_ = ireeCompilerInvocationCreate(session_);
    if (!inv_) {
        ireeCompilerSessionDestroy(session_);
        throw std::runtime_error("Failed to create IREE compiler invocation");
    }
    ireeCompilerInvocationEnableConsoleDiagnostics(inv_);
}

conquer::IREECompilerSession::~IREECompilerSession() {
    L_DEBUG("Destroying IREE compiler session and invocation.");
    if (output_)
        ireeCompilerOutputDestroy(output_);
    if (source_)
        ireeCompilerSourceDestroy(source_);
    if (inv_)
        ireeCompilerInvocationDestroy(inv_);
    if (session_)
        ireeCompilerSessionDestroy(session_);
}

conquer::IREECompilerSession::IREECompilerSession(IREECompilerSession &&other) noexcept
    : session_(other.session_), inv_(other.inv_), source_(other.source_), output_(other.output_) {
    other.session_ = nullptr;
    other.inv_ = nullptr;
    other.source_ = nullptr;
    other.output_ = nullptr;
}

conquer::IREECompilerSession &conquer::IREECompilerSession::operator=(IREECompilerSession &&other) noexcept {
    if (this != &other) {
        this->~IREECompilerSession();

        session_ = other.session_;
        inv_ = other.inv_;
        source_ = other.source_;
        output_ = other.output_;

        other.session_ = nullptr;
        other.inv_ = nullptr;
        other.source_ = nullptr;
        other.output_ = nullptr;
    }
    return *this;
}

llvm::Error conquer::IREECompilerSession::wrap_source_buffer(const std::string &name, const std::string &buffer) {
    L_TRACE("Wrapping source buffer for IREE compiler session.");
    if (source_) {
        ireeCompilerSourceDestroy(source_);
        source_ = nullptr;
    }

    iree_compiler_error_t *error =
        ireeCompilerSourceWrapBuffer(session_, name.c_str(), buffer.c_str(), buffer.size() + 1,
                                     /*isNullTerminated=*/true, &source_);

    if (error)
        return make_llvm_error(error);

    if (!ireeCompilerInvocationParseSource(inv_, source_)) {
        return llvm::make_error<llvm::StringError>("Error parsing input source into invocation",
                                                   llvm::inconvertibleErrorCode());
    }

    return llvm::Error::success();
}


llvm::Error conquer::IREECompilerSession::wrap_module_bytecode(llvm::StringRef bytecode) {
    if (!bytecode.data() || bytecode.empty()) {
        return llvm::make_error<llvm::StringError>("Cannot serialise null MLIR module", llvm::inconvertibleErrorCode());
    }

    if (source_) {
        ireeCompilerSourceDestroy(source_);
        source_ = nullptr;
    }

    iree_compiler_source_t *local_source = nullptr;
    if (iree_compiler_error_t *error =
            ireeCompilerSourceWrapBuffer(session_, "module.mlirbc", bytecode.data(), bytecode.size(),
                                         /*isNullTerminated=*/false, &local_source)) {
        return make_llvm_error(error);
                                         }

    auto cleanup = llvm::scope_exit([&] {
        if (local_source) {
            ireeCompilerSourceDestroy(local_source);
        }
    });

    if (!ireeCompilerInvocationParseSource(inv_, local_source)) {
        return llvm::make_error<llvm::StringError>("Error parsing MLIR bytecode into IREE invocation",
                                                   llvm::inconvertibleErrorCode());
    }

    return llvm::Error::success();
}

llvm::Error conquer::IREECompilerSession::wrap_module_bytecode(const std::string &name, mlir::Operation *module) {
    L_TRACE("Serialising MLIR module to bytecode for IREE compiler session.");

    if (!module) {
        return llvm::make_error<llvm::StringError>("Cannot serialise null MLIR module", llvm::inconvertibleErrorCode());
    }

    if (source_) {
        ireeCompilerSourceDestroy(source_);
        source_ = nullptr;
    }

    llvm::SmallVector<char, 0> bytecode_storage;
    llvm::raw_svector_ostream os(bytecode_storage);

    if (mlir::failed(mlir::writeBytecodeToFile(module, os))) {
        return llvm::make_error<llvm::StringError>("Failed to serialize MLIR module to bytecode",
                                                   llvm::inconvertibleErrorCode());
    }

    iree_compiler_source_t *local_source = nullptr;
    if (iree_compiler_error_t *error =
            ireeCompilerSourceWrapBuffer(session_, name.c_str(), bytecode_storage.data(), bytecode_storage.size(),
                                         /*isNullTerminated=*/false, &local_source)) {
        return make_llvm_error(error);
    }

    auto cleanup = llvm::scope_exit([&] {
        if (local_source) {
            ireeCompilerSourceDestroy(local_source);
        }
    });

    if (!ireeCompilerInvocationParseSource(inv_, local_source)) {
        return llvm::make_error<llvm::StringError>("Error parsing MLIR bytecode into IREE invocation",
                                                   llvm::inconvertibleErrorCode());
    }

    return llvm::Error::success();
}

llvm::Error conquer::IREECompilerSession::import_steal_module(mlir::OwningOpRef<mlir::ModuleOp> module) {
    L_TRACE("Importing MLIR module into IREE compiler invocation via steal.");

    if (!module) {
        return llvm::make_error<llvm::StringError>("Cannot import null MLIR module", llvm::inconvertibleErrorCode());
    }

    if (source_) {
        ireeCompilerSourceDestroy(source_);
        source_ = nullptr;
    }

    mlir::ModuleOp released_module = module.release();

    if (!ireeCompilerInvocationImportStealModule(inv_, MlirOperation{released_module.getOperation()})) {
        return llvm::make_error<llvm::StringError>(
            "Error importing MLIR module into IREE invocation (see console diagnostics)",
            llvm::inconvertibleErrorCode());
    }

    return llvm::Error::success();
}

llvm::Error conquer::IREECompilerSession::import_borrow_module(mlir::Operation *module) {
    L_TRACE("Importing MLIR module into IREE compiler invocation via borrow.");

    if (!module) {
        return llvm::make_error<llvm::StringError>("Cannot import null MLIR module", llvm::inconvertibleErrorCode());
    }

    if (source_) {
        ireeCompilerSourceDestroy(source_);
        source_ = nullptr;
    }

    if (!ireeCompilerInvocationImportBorrowModule(inv_, MlirOperation{module})) {
        return llvm::make_error<llvm::StringError>(
            "Error importing MLIR module into IREE invocation (see console diagnostics)",
            llvm::inconvertibleErrorCode());
    }

    return llvm::Error::success();
}

llvm::Error conquer::IREECompilerSession::open_output_membuffer() {
    L_TRACE("Opening output membuffer for IREE compiler session.");
    if (output_) {
        ireeCompilerOutputDestroy(output_);
        output_ = nullptr;
    }

    if (iree_compiler_error_t *error = ireeCompilerOutputOpenMembuffer(&output_))
        return make_llvm_error(error);

    return llvm::Error::success();
}

llvm::Error conquer::IREECompilerSession::set_flags(const std::vector<std::string> &flags) const {
    L_DEBUG("Setting IREE compiler flags:");
    for (const auto &flag : flags) {
        L_TRACE("  " << flag);
    }
    std::vector<const char *> c_flags;
    c_flags.reserve(flags.size());
    for (const auto &flag : flags) {
        c_flags.push_back(flag.c_str());
    }

    iree_compiler_error_t *error =
        ireeCompilerSessionSetFlags(session_, static_cast<int>(c_flags.size()), c_flags.data());

    return make_llvm_error(error);
}

llvm::Expected<std::vector<uint8_t>> conquer::IREECompilerSession::extract_output() const {
    L_DEBUG("Extracting output from IREE compiler session.");
    if (!output_) {
        return llvm::make_error<llvm::StringError>("No output buffer to extract from", llvm::inconvertibleErrorCode());
    }

    void *contents = nullptr;
    uint64_t size = 0;

    if (iree_compiler_error_t *error = ireeCompilerOutputMapMemory(output_, &contents, &size)) {
        return make_llvm_error(error);
    }

    // Copy the mapped memory into your C++ vector so the compiler state can be destroyed
    auto *bytes = static_cast<const uint8_t *>(contents);
    return std::vector(bytes, bytes + size);
}

llvm::Error conquer::make_llvm_error(iree_compiler_error_t *iree_error) {
    if (!iree_error) {
        return llvm::Error::success();
    }
    std::string message = ireeCompilerErrorGetMessage(iree_error);
    ireeCompilerErrorDestroy(iree_error);
    return llvm::make_error<llvm::StringError>(message, llvm::inconvertibleErrorCode());
}

iree_hal_element_type_t conquer::mapConquerTypeToIREE(const DataType type) {
    switch (type) {
    case DataType::FP32:
        return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case DataType::FP16:
        return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case DataType::BF16:
        return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
    case DataType::FP8_E4M3:
        return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN;
    case DataType::FP8_E5M2:
        return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2;
    case DataType::INT32:
        return IREE_HAL_ELEMENT_TYPE_SINT_32;
    case DataType::INT16:
        return IREE_HAL_ELEMENT_TYPE_SINT_16;
    // case DataType::UINT8:
    //     return IREE_HAL_ELEMENT_TYPE_UINT_8;
    case DataType::INT8:
        return IREE_HAL_ELEMENT_TYPE_SINT_8;
    // case DataType::UINT4:
    //     return IREE_HAL_ELEMENT_TYPE_UINT_4;
    case DataType::INT4:
        return IREE_HAL_ELEMENT_TYPE_SINT_4;
    default:
        return IREE_HAL_ELEMENT_TYPE_NONE;
    }
}

// Runtime
conquer::IREERuntimeSession::IREERuntimeSession(const std::vector<uint8_t> &module, const IREETargetBackend backend,
                                                const std::string &device_uri)
    : session_(nullptr) {
    LLVM_DEBUG(llvm::dbgs() << "Creating IREE runtime session for backend: " << to_string(backend) << ".\n");
    iree_string_view_t driver_name = iree_make_cstring_view("local-task");

    switch (backend) {
    case IREETargetBackend::VULKAN:
        driver_name = iree_make_cstring_view("vulkan");
        break;
    case IREETargetBackend::CPU:
        driver_name = iree_make_cstring_view("local-task");
        break;
    case IREETargetBackend::CUDA:
        driver_name = iree_make_cstring_view("cuda");
        break;
    case IREETargetBackend::ROCM:
        driver_name = iree_make_cstring_view("rocm");
        break;
    }
    iree_hal_device_t *device = nullptr;
    if (!device_uri.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "Attempting to create IREE device from URI: " << device_uri << ".\n");
        const iree_string_view_t uri_view = iree_make_string_view(device_uri.c_str(), device_uri.size());

        IREE_CHECK_OK(
            iree_hal_create_device(iree_runtime_instance_driver_registry(get_iree_instance().instance), uri_view,
                                   iree_runtime_instance_host_allocator(get_iree_instance().instance), &device));
    } else {
        LLVM_DEBUG(llvm::dbgs() << "No device URI provided, falling back to default device for driver.\n");
        IREE_CHECK_OK(
            iree_runtime_instance_try_create_default_device(get_iree_instance().instance, driver_name, &device));
    }

    LLVM_DEBUG(llvm::dbgs() << "Successfully created IREE device for backend: " << to_string(backend) << ".\n");

    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    IREE_CHECK_OK(iree_runtime_session_create_with_device(
        get_iree_instance().instance, &session_options, device,
        iree_runtime_instance_host_allocator(get_iree_instance().instance), &session_));
    iree_hal_device_release(device);

    LLVM_DEBUG(llvm::dbgs() << "Successfully created IREE session for backend: " << to_string(backend) << ".\n");

    IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_memory(
        session_, iree_make_const_byte_span(module.data(), module.size()), iree_allocator_null()));

    LLVM_DEBUG(llvm::dbgs() << "Successfully loaded module into IREE session for backend: " << to_string(backend)
                            << ".\n");
}

conquer::IREERuntimeSession::~IREERuntimeSession() {
    L_DEBUG("Releasing IREE runtime session.");
    if (session_) {
        iree_runtime_session_release(session_);
        session_ = nullptr;
    }
}

conquer::IREERuntimeSession::IREERuntimeSession(IREERuntimeSession &&other) noexcept
    : session_(other.session_), call(other.call) {
    other.session_ = nullptr;
    other.call = {};
}

conquer::IREERuntimeSession &conquer::IREERuntimeSession::operator=(IREERuntimeSession &&other) noexcept {
    if (this != &other) {
        this->~IREERuntimeSession();
        session_ = other.session_;
        other.session_ = nullptr;
    }
    return *this;
}

void conquer::IREERuntimeSession::invoke(const std::vector<TensorView> &inputs,
                                         const std::vector<TensorView> &outputs) {
    IREE_CHECK_OK(iree_runtime_call_initialize_by_name(session_, iree_make_cstring_view("module.main"), &call));

    auto cleanup = llvm::scope_exit([&] {
        iree_runtime_call_deinitialize(&call);
    });

    prepare_inputs(inputs);
    LLVM_DEBUG(llvm::dbgs() << "Invoking IREE runtime session.\n");
    IREE_CHECK_OK(iree_runtime_call_invoke(&call, /*flags=*/0));
    LLVM_DEBUG(llvm::dbgs() << "Successfully invoked IREE runtime session.\n");
    prepare_outputs(outputs);
}

void conquer::IREERuntimeSession::prepare_inputs(const std::vector<TensorView> &inputs) {
    L_TRACE("Preparing inputs for IREE runtime session.");
    for (const auto &input : inputs) {
        const iree_hal_element_type_t iree_type = mapConquerTypeToIREE(input.dtype);
        if (iree_type == IREE_HAL_ELEMENT_TYPE_NONE) {
            throw std::runtime_error("Unsupported conquer::DataType encountered during IREE conversion.");
        }

        std::vector<iree_hal_dim_t> iree_shape(input.shape.begin(), input.shape.end());
        const size_t byte_size = input.getNumElements() * get_size_bytes(input.dtype);

        iree_hal_buffer_params_t params = {};

        // Transfers required to fetch outputs off the device (if not cpu) and back to host.
        params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                       IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
        params.access = IREE_HAL_MEMORY_ACCESS_ALL;
        params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
        params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
        params.min_alignment = 0;

        iree_hal_buffer_view_t *buffer_view = nullptr;
        IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
            iree_runtime_session_device(session_), iree_runtime_session_device_allocator(session_), iree_shape.size(),
            iree_shape.data(), iree_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
            iree_make_const_byte_span(input.data, byte_size), &buffer_view));

        IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, buffer_view));

        iree_hal_buffer_view_release(buffer_view);
    }
    L_TRACE("Finished preparing inputs for IREE runtime session.");
}

void conquer::IREERuntimeSession::prepare_outputs(const std::vector<TensorView> &outputs) {
    L_TRACE("Fetching outputs for IREE runtime session.");
    for (size_t i = 0; i < outputs.size(); ++i) {
        iree_hal_buffer_view_t *buffer_view = nullptr;
        IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &buffer_view));

        const auto &output = outputs[i];
        const size_t byte_size = output.getNumElements() * get_size_bytes(output.dtype);

        IREE_CHECK_OK(iree_hal_device_transfer_d2h(iree_runtime_session_device(session_),
                                                   iree_hal_buffer_view_buffer(buffer_view),
                                                   /*source_offset=*/0, output.data, byte_size,
                                                   IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

        L_TRACE("Successfully copied D2H into outputs[" << i << "]");

        iree_hal_buffer_view_release(buffer_view);
    }
    L_TRACE("Finished fetching outputs for IREE runtime session.");
}
