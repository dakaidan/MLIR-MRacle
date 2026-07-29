#pragma once

#include "conquer/backend/capabilities.h"
#include "conquer/backend/target.h"

#include <vector>
#include <memory>

#include "iree_wrapper.h"

namespace conquer {
  struct IREERuntimeState;

  class IREETarget : public Target {
  public:
    explicit IREETarget(IREETargetBackend backend);
    ~IREETarget() = default;

    IREETarget(const IREETarget &) = delete;
    IREETarget &operator=(const IREETarget &) = delete;

    IREETarget(IREETarget &&other) noexcept;
    IREETarget &operator=(IREETarget &&other) noexcept;

    llvm::Error compile(mlir::Operation *module) override;
    llvm::Error compile(llvm::StringRef bytecode) override;
    llvm::Error execute(const std::vector<TensorView> &inputs, const std::vector<TensorView> &outputs) override;

    llvm::Error save_compiled_module_to_file(const std::string &filename) const override;
    llvm::Error load_compiled_module_from_file(const std::string &filename) override;

    [[nodiscard]] HardwareCapability query_capability() const override;

    IREETargetBackend get_backend_type() const { return backend_; }

  protected:
    IREETargetBackend backend_;
    std::vector<uint8_t> compiled_module_data_;
    HardwareCapability target_capability;
    std::unique_ptr<IREERuntimeSession> runtime_session_;
  };

  template <IREETargetBackend Backend> class IREEBackendTarget : public IREETarget {
  public:
    IREEBackendTarget() : IREETarget(Backend) {}
  };
} // namespace conquer