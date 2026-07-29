#include <gtest/gtest.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "conquer/core/context.h"
#include "conquer/core/types.h"
#include "conquer/dialect/ConquerDialect.h"
#include "conquer/quantisation/policy.h"
#include "conquer/quantisation/runner.h"
#include "conquer/passes/naming_pass.h"

class ConquerQuantisationTest : public ::testing::Test {
protected:
    std::shared_ptr<mlir::MLIRContext> context;

    void SetUp() override {
        context = conquer::createMLIRContext();
        context->getOrLoadDialect<mlir::tosa::TosaDialect>();
        context->getOrLoadDialect<mlir::func::FuncDialect>();
        context->getOrLoadDialect<conquer::ConquerDialect>();
    }

    // -------------------------------------------------------------------------
    // Type helpers
    // -------------------------------------------------------------------------
    [[nodiscard]] mlir::FloatType   f32() const { return mlir::Float32Type::get(context.get()); }
    [[nodiscard]] mlir::IntegerType i32() const { return mlir::IntegerType::get(context.get(), 32); }
    [[nodiscard]] mlir::IntegerType i8()  const { return mlir::IntegerType::get(context.get(), 8);  }
    [[nodiscard]] mlir::IntegerType i1()  const { return mlir::IntegerType::get(context.get(), 1);  }

    static mlir::RankedTensorType tensorOf(const mlir::Type elem, const llvm::ArrayRef<int64_t> shape) {
        return mlir::RankedTensorType::get(shape, elem);
    }

    // -------------------------------------------------------------------------
    // Constant builders
    // -------------------------------------------------------------------------
    static mlir::Value floatConst(mlir::OpBuilder& b, const mlir::Location loc,
                           const mlir::RankedTensorType ty, const float val) {
        int64_t num_elems = ty.getNumElements();
        std::vector<float> values;
        values.reserve(num_elems);

        // Prevent splats for weights (val != 0) to ensure a non-zero range for quantisation scaling
        if (num_elems == 1 || val == 0.0f) {
            for (int64_t i = 0; i < num_elems; ++i) {
                values.push_back(val);
            }
        } else {
            for (int64_t i = 0; i < num_elems; ++i) {
                float fraction = static_cast<float>(i) / (num_elems - 1);
                values.push_back(val * (2.0f * fraction - 1.0f)); // Range [-val, val]
            }
        }

        const auto attr = mlir::DenseElementsAttr::get(ty, llvm::ArrayRef<float>(values));
        return mlir::tosa::ConstOp::create(b, loc, ty, attr).getResult();
    }

    static mlir::Value intConst(mlir::OpBuilder& b, const mlir::Location loc,
                                const mlir::RankedTensorType ty, const int64_t val) {
        std::vector<int64_t> values(ty.getNumElements(), val);
        const auto attr = mlir::DenseElementsAttr::get(ty, llvm::ArrayRef<int64_t>(values));
        return mlir::tosa::ConstOp::create(b, loc, ty, attr).getResult();
    }

    static mlir::Value castTensor(mlir::OpBuilder& b, const mlir::Location loc,
                                  const mlir::RankedTensorType dstTy, const mlir::Value src) {
        return mlir::tosa::CastOp::create(b, loc, dstTy, src).getResult();
    }

    // tensor<1xf32> zero-point — avoids 0-d tensor portability issues.
    mlir::Value zpF32(mlir::OpBuilder& b, const mlir::Location loc) const {
        return floatConst(b, loc, tensorOf(f32(), {1}), 0.0f);
    }

    // tensor<1xi8> shift for MulOp.
    mlir::Value shiftI8(mlir::OpBuilder& b, const mlir::Location loc) const {
        auto f32Ty = tensorOf(f32(), {1});
        auto i8Ty = tensorOf(i8(), {1});
        auto zeroF32 = floatConst(b, loc, f32Ty, 0.0f);
        return castTensor(b, loc, i8Ty, zeroF32);
    }

    // -------------------------------------------------------------------------
    // Module factory
    // -------------------------------------------------------------------------
    using BodyFn = std::function<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;
    using ModuleFactory = std::function<mlir::OwningOpRef<mlir::ModuleOp>()>;

    [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp> makeModule(
        const mlir::TypeRange inputTypes, const mlir::TypeRange outputTypes, const BodyFn& body) const {
        const auto loc = mlir::UnknownLoc::get(context.get());
        mlir::OwningOpRef<mlir::ModuleOp> mod(mlir::ModuleOp::create(loc));
        mlir::OpBuilder b(context.get());
        b.setInsertionPointToEnd(mod->getBody());
        const auto funcType = mlir::FunctionType::get(context.get(), inputTypes, outputTypes);
        auto func = mlir::func::FuncOp::create(b, loc, "main", funcType);
        auto* block = func.addEntryBlock();
        b.setInsertionPointToStart(block);
        body(b, loc, block->getArguments());
        return mod;
    }

    static void assertMainBoundaryIsFp32(mlir::ModuleOp module) {
        auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
        ASSERT_TRUE(mainFunc) << "Missing main function.";

        for (auto type : mainFunc.getArgumentTypes()) {
            auto rankedTy = mlir::dyn_cast<mlir::RankedTensorType>(type);
            ASSERT_TRUE(rankedTy) << "Only ranked tensor arguments are supported in this test.";
            ASSERT_TRUE(rankedTy.getElementType().isF32()) << "All inputs must be fp32.";
        }

        for (auto type : mainFunc.getResultTypes()) {
            auto rankedTy = mlir::dyn_cast<mlir::RankedTensorType>(type);
            ASSERT_TRUE(rankedTy) << "Only ranked tensor results are supported in this test.";
            ASSERT_TRUE(rankedTy.getElementType().isF32()) << "All outputs must be fp32.";
        }
    }

    static std::vector<conquer::TensorAllocation> buildFp32CalibrationForMain(mlir::func::FuncOp mainFunc) {
        std::vector<conquer::TensorAllocation> calibration;
        calibration.reserve(mainFunc.getNumArguments());

        for (auto type : mainFunc.getArgumentTypes()) {
            auto rankedTy = mlir::dyn_cast<mlir::RankedTensorType>(type);

            conquer::TensorAllocation alloc;
            alloc.shape = rankedTy.getShape().vec();
            alloc.dtype = conquer::DataType::FP32;

            const size_t numElems = rankedTy.getNumElements();
            alloc.buffer.resize(numElems * sizeof(float));
            auto *ptr = reinterpret_cast<float *>(alloc.buffer.data());
            for (size_t i = 0; i < numElems; ++i) {
                ptr[i] = static_cast<float>(i + 1);
            }

            calibration.push_back(std::move(alloc));
        }

        return calibration;
    }

    static void testAllSingleOpConfigs(const ModuleFactory& factory) {
        auto probe = factory();
        ASSERT_TRUE(probe) << "Module construction failed.";
        assertMainBoundaryIsFp32(probe.get());
        conquer::runPassOnModule<conquer::ModelNamingPass>(probe.get());

        auto policy = conquer::generate_default_policy(probe.get());
        ASSERT_FALSE(policy.layers.empty()) << "No quantisable ops found.";

        const std::string target = policy.layers.begin()->first;
        auto mutations = conquer::generate_mutations(target);
        ASSERT_FALSE(mutations.empty()) << "generate_mutations returned 0 configs!";

        int count = 0;

        llvm::outs() << "[INFO] Testing " << mutations.size() << " mutations for target op: " << target << "\n";

        for (const auto& mutation : mutations) {
            auto mod = factory();
            ASSERT_TRUE(mod);
            assertMainBoundaryIsFp32(mod.get());
            conquer::runPassOnModule<conquer::ModelNamingPass>(mod.get());
            auto mainFunc = mod.get().lookupSymbol<mlir::func::FuncOp>("main");
            ASSERT_TRUE(mainFunc) << "Missing main function.";
            auto dummy_calib = buildFp32CalibrationForMain(mainFunc);

            conquer::QuantisationPolicy test_policy;
            test_policy.layers[target] = mutation;

            EXPECT_NO_THROW({
                // Rebuild calibration inputs for each mutation and keep test I/O fp32.
                conquer::quantise(mod.get(), test_policy, dummy_calib);
                EXPECT_TRUE(mod.get().verify().succeeded())
                    << "Verification failed for: " << target;
            }) << "Crash on: " << target;
            ++count;
        }
        llvm::outs() << "[INFO] Tested " << count << " mutations for " << target << "\n";
    }

    // -------------------------------------------------------------------------
    // Elementwise / reduction factory templates
    // -------------------------------------------------------------------------

    // Binary f32→f32, two activations.
    template <typename OpType>
    ModuleFactory binaryF32ActAct() {
        llvm::outs() << "Creating binaryF32ActAct factory for " << OpType::getOperationName() << "\n";
        llvm::outs().flush();
        return [this] {
            auto ty = tensorOf(f32(), {4, 4});
            return makeModule({ty}, {ty},
                [ty](mlir::OpBuilder& b, mlir::Location loc, const mlir::ValueRange args) {
                    auto out = OpType::create(b, loc, ty, args[0], args[0]);
                    mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out.getResult()});
                });
        };
    }

    // Binary f32→f32, activation + constant.
    template <typename OpType>
    ModuleFactory binaryF32ActConst() {
        return [this] {
            auto ty = tensorOf(f32(), {4, 4});
            return makeModule({ty}, {ty},
                [ty](mlir::OpBuilder& b, mlir::Location loc, const mlir::ValueRange args) {
                    auto w   = floatConst(b, loc, ty, 1.0f);
                    auto out = OpType::create(b, loc, ty, args[0], w);
                    mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out.getResult()});
                });
        };
    }

    // maximum / minimum: require nan_mode attribute.
    template <typename OpType>
    ModuleFactory minMaxActAct() {
        return [this] {
            auto ty = tensorOf(f32(), {4, 4});
            return makeModule({ty}, {ty},
                [this, ty](mlir::OpBuilder& b, mlir::Location loc, const mlir::ValueRange args) {
                    auto nm = mlir::tosa::NanPropagationModeAttr::get(
                        context.get(), mlir::tosa::NanPropagationMode::PROPAGATE);
                    auto out = OpType::create(b, loc, ty, args[0], args[0], nm);
                    mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out.getResult()});
                });
        };
    }

    template <typename OpType>
    ModuleFactory minMaxActConst() {
        return [this] {
            auto ty = tensorOf(f32(), {4, 4});
            return makeModule({ty}, {ty},
                [this, ty](mlir::OpBuilder& b, mlir::Location loc, const mlir::ValueRange args) {
                    auto w  = floatConst(b, loc, ty, 1.0f);
                    auto nm = mlir::tosa::NanPropagationModeAttr::get(
                        context.get(), mlir::tosa::NanPropagationMode::PROPAGATE);
                    auto out = OpType::create(b, loc, ty, args[0], w, nm);
                    mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out.getResult()});
                });
        };
    }

    // Comparison: f32 in/out, i1 intermediate.
    template <typename OpType>
    ModuleFactory cmpActAct() {
        return [this] {
            auto inTy  = tensorOf(f32(), {4, 4});
            auto cmpTy = tensorOf(i1(), {4, 4});
            auto outTy = tensorOf(f32(), {4, 4});
            return makeModule({inTy}, {outTy},
                [outTy, cmpTy](mlir::OpBuilder& b, mlir::Location loc, const mlir::ValueRange args) {
                    auto cmp = OpType::create(b, loc, cmpTy, args[0], args[0]);
                    auto out = castTensor(b, loc, outTy, cmp.getResult());
                    mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out});
                });
        };
    }

    template <typename OpType>
    ModuleFactory cmpActConst() {
        return [this] {
            auto inTy  = tensorOf(f32(), {4, 4});
            auto cmpTy = tensorOf(i1(), {4, 4});
            auto outTy = tensorOf(f32(), {4, 4});
            return makeModule({inTy}, {outTy},
                [inTy, outTy, cmpTy](mlir::OpBuilder& b, mlir::Location loc, const mlir::ValueRange args) {
                    auto w = floatConst(b, loc, inTy, 1.0f);
                    auto cmp = OpType::create(b, loc, cmpTy, args[0], w);
                    auto out = castTensor(b, loc, outTy, cmp.getResult());
                    mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out});
                });
        };
    }

    // Reduction f32, axis=0, no nan_mode.
    template <typename OpType>
    ModuleFactory reductionF32() {
        return [this] {
            auto inTy  = tensorOf(f32(), {4, 4});
            auto outTy = tensorOf(f32(), {1, 4});
            return makeModule({inTy}, {outTy},
                [outTy](mlir::OpBuilder& b, mlir::Location loc, const mlir::ValueRange args) {
                    auto out = OpType::create(b, loc, outTy, args[0], b.getI32IntegerAttr(0));
                    mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out.getResult()});
                });
        };
    }

    // Reduction f32, axis=0, with nan_mode.
    template <typename OpType>
    ModuleFactory reductionNanMode() {
        return [this] {
            auto inTy  = tensorOf(f32(), {4, 4});
            auto outTy = tensorOf(f32(), {1, 4});
            return makeModule({inTy}, {outTy},
                [this, outTy](mlir::OpBuilder& b, mlir::Location loc, const mlir::ValueRange args) {
                    auto nm = mlir::tosa::NanPropagationModeAttr::get(
                        context.get(), mlir::tosa::NanPropagationMode::PROPAGATE);
                    auto out = OpType::create(
                        b, loc, outTy, args[0], b.getI32IntegerAttr(0), nm);
                    mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out.getResult()});
                });
        };
    }
};

// =============================================================================
// Group 1: Accumulating Tensor Ops
// =============================================================================

TEST_F(ConquerQuantisationTest, Conv2DOp) {
    testAllSingleOpConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 4, 4, 3});
        auto outTy = tensorOf(f32(), {1, 2, 2, 8});
        return makeModule({inTy}, {outTy},
            [this, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w    = floatConst(b, loc, tensorOf(f32(), {8, 3, 3, 3}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {8}), 1.0f);
                const auto izp  = zpF32(b, loc);
                const auto wzp  = zpF32(b, loc);
                auto conv = mlir::tosa::Conv2DOp::create(
                    b, loc, outTy, args[0], w, bias, izp, wzp,
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}),  // pad
                    b.getDenseI64ArrayAttr({1, 1}),         // stride
                    b.getDenseI64ArrayAttr({1, 1}),         // dilation
                    mlir::TypeAttr::get(f32()),              // acc_type
                    b.getBoolAttr(false));                   // local_bound
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{conv.getResult()});
            });
    });
}

TEST_F(ConquerQuantisationTest, Conv3DOp) {
    testAllSingleOpConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 4, 4, 4, 3});
        auto outTy = tensorOf(f32(), {1, 2, 2, 2, 8});
        return makeModule({inTy}, {outTy},
            [this, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w    = floatConst(b, loc, tensorOf(f32(), {8, 3, 3, 3, 3}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {8}), 1.0f);
                const auto izp  = zpF32(b, loc);
                const auto wzp  = zpF32(b, loc);
                auto conv = mlir::tosa::Conv3DOp::create(
                    b, loc, outTy, args[0], w, bias, izp, wzp,
                    b.getDenseI64ArrayAttr({0, 0, 0, 0, 0, 0}),  // pad (6)
                    b.getDenseI64ArrayAttr({1, 1, 1}),             // stride (3)
                    b.getDenseI64ArrayAttr({1, 1, 1}),             // dilation (3)
                    mlir::TypeAttr::get(f32()),
                    b.getBoolAttr(false));
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{conv.getResult()});
            });
    });
}

TEST_F(ConquerQuantisationTest, DepthwiseConv2DOp) {
    testAllSingleOpConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 4, 4, 3});
        auto outTy = tensorOf(f32(), {1, 2, 2, 6});
        return makeModule({inTy}, {outTy},
            [this, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w    = floatConst(b, loc, tensorOf(f32(), {3, 3, 3, 2}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {6}), 1.0f);
                const auto izp  = zpF32(b, loc);
                const auto wzp  = zpF32(b, loc);
                auto conv = mlir::tosa::DepthwiseConv2DOp::create(
                    b, loc, outTy, args[0], w, bias, izp, wzp,
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}),
                    b.getDenseI64ArrayAttr({1, 1}),
                    b.getDenseI64ArrayAttr({1, 1}),
                    mlir::TypeAttr::get(f32()),
                    b.getBoolAttr(false));
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{conv.getResult()});
            });
    });
}

TEST_F(ConquerQuantisationTest, TransposeConv2DOp) {
    testAllSingleOpConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 2, 2, 3});
        // Correct output shape resolved: OH = (IH - 1) * stride_y + out_pad_top + out_pad_bottom + KH -> (2-1)*2 + 3 = 5
        auto outTy = tensorOf(f32(), {1, 5, 5, 8});
        return makeModule({inTy}, {outTy},
            [this, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w    = floatConst(b, loc, tensorOf(f32(), {8, 3, 3, 3}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {8}), 1.0f);
                const auto izp  = zpF32(b, loc);
                const auto wzp  = zpF32(b, loc);
                auto conv = mlir::tosa::TransposeConv2DOp::create(
                    b, loc, outTy, args[0], w, bias, izp, wzp,
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}),  // out_pad
                    b.getDenseI64ArrayAttr({2, 2}),         // stride
                    mlir::TypeAttr::get(f32()),
                    b.getBoolAttr(false));
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{conv.getResult()});
            });
    });
}

TEST_F(ConquerQuantisationTest, MatMulOp) {
    testAllSingleOpConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 4, 8});
        auto outTy = tensorOf(f32(), {1, 4, 16});
        return makeModule({inTy}, {outTy},
            [this, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w   = floatConst(b, loc, tensorOf(f32(), {1, 8, 16}), 1.0f);
                const auto azp = zpF32(b, loc);
                const auto bzp = zpF32(b, loc);
                auto mm  = mlir::tosa::MatMulOp::create(
                    b, loc, outTy, args[0], w, azp, bzp);
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{mm.getResult()});
            });
    });
}

// =============================================================================
// Group 2: Elementwise Binary Math Ops
// =============================================================================

TEST_F(ConquerQuantisationTest, Elementwise_add_ActAct) {
    testAllSingleOpConfigs(binaryF32ActAct<mlir::tosa::AddOp>());
}
TEST_F(ConquerQuantisationTest, Elementwise_add_ActConst)
    { testAllSingleOpConfigs(binaryF32ActConst<mlir::tosa::AddOp>()); }

TEST_F(ConquerQuantisationTest, Elementwise_sub_ActAct)
    { testAllSingleOpConfigs(binaryF32ActAct<mlir::tosa::SubOp>()); }
TEST_F(ConquerQuantisationTest, Elementwise_sub_ActConst)
    { testAllSingleOpConfigs(binaryF32ActConst<mlir::tosa::SubOp>()); }

TEST_F(ConquerQuantisationTest, Elementwise_pow_ActAct)
    { testAllSingleOpConfigs(binaryF32ActAct<mlir::tosa::PowOp>()); }
TEST_F(ConquerQuantisationTest, Elementwise_pow_ActConst)
    { testAllSingleOpConfigs(binaryF32ActConst<mlir::tosa::PowOp>()); }

// mul requires an explicit shift operand.
TEST_F(ConquerQuantisationTest, Elementwise_mul_ActAct) {
    testAllSingleOpConfigs([this] {
        auto ty = tensorOf(f32(), {4, 4});
        return makeModule({ty}, {ty},
            [this, ty](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                auto out = mlir::tosa::MulOp::create(
                    b, loc, ty, args[0], args[0], shiftI8(b, loc));
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out.getResult()});
            });
    });
}
TEST_F(ConquerQuantisationTest, Elementwise_mul_ActConst) {
    testAllSingleOpConfigs([this] {
        auto ty = tensorOf(f32(), {4, 4});
        return makeModule({ty}, {ty},
            [this, ty](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w   = floatConst(b, loc, ty, 1.0f);
                auto out = mlir::tosa::MulOp::create(
                    b, loc, ty, args[0], w, shiftI8(b, loc));
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{out.getResult()});
            });
    });
}

TEST_F(ConquerQuantisationTest, Elementwise_maximum_ActAct)
    { testAllSingleOpConfigs(minMaxActAct<mlir::tosa::MaximumOp>()); }
TEST_F(ConquerQuantisationTest, Elementwise_maximum_ActConst)
    { testAllSingleOpConfigs(minMaxActConst<mlir::tosa::MaximumOp>()); }

TEST_F(ConquerQuantisationTest, Elementwise_minimum_ActAct)
    { testAllSingleOpConfigs(minMaxActAct<mlir::tosa::MinimumOp>()); }
TEST_F(ConquerQuantisationTest, Elementwise_minimum_ActConst)
    { testAllSingleOpConfigs(minMaxActConst<mlir::tosa::MinimumOp>()); }

TEST_F(ConquerQuantisationTest, Elementwise_equal_ActAct)
    { testAllSingleOpConfigs(cmpActAct<mlir::tosa::EqualOp>()); }
TEST_F(ConquerQuantisationTest, Elementwise_equal_ActConst)
    { testAllSingleOpConfigs(cmpActConst<mlir::tosa::EqualOp>()); }

TEST_F(ConquerQuantisationTest, Elementwise_greater_ActAct)
    { testAllSingleOpConfigs(cmpActAct<mlir::tosa::GreaterOp>()); }
TEST_F(ConquerQuantisationTest, Elementwise_greater_ActConst)
    { testAllSingleOpConfigs(cmpActConst<mlir::tosa::GreaterOp>()); }

TEST_F(ConquerQuantisationTest, Elementwise_greater_equal_ActAct)
    { testAllSingleOpConfigs(cmpActAct<mlir::tosa::GreaterEqualOp>()); }
TEST_F(ConquerQuantisationTest, Elementwise_greater_equal_ActConst)
    { testAllSingleOpConfigs(cmpActConst<mlir::tosa::GreaterEqualOp>()); }

// arithmetic_right_shift: integer-only, requires round attribute.
TEST_F(ConquerQuantisationTest, Elementwise_arithmetic_right_shift_ActAct) {
    testAllSingleOpConfigs([this] {
        auto ioTy = tensorOf(f32(), {4, 4});
        auto opTy = tensorOf(i32(), {4, 4});
        return makeModule({ioTy}, {ioTy},
            [ioTy, opTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                auto lhs = castTensor(b, loc, opTy, args[0]);
                auto rhs = castTensor(b, loc, opTy, args[0]);
                auto out = mlir::tosa::ArithmeticRightShiftOp::create(
                    b, loc, opTy, lhs, rhs, b.getBoolAttr(false));
                auto outF32 = castTensor(b, loc, ioTy, out.getResult());
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{outF32});
            });
    });
}

TEST_F(ConquerQuantisationTest, Elementwise_arithmetic_right_shift_ActConst) {
    testAllSingleOpConfigs([this] {
        auto ioTy = tensorOf(f32(), {4, 4});
        auto opTy = tensorOf(i32(), {4, 4});
        return makeModule({ioTy}, {ioTy},
            [ioTy, opTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                auto lhs = castTensor(b, loc, opTy, args[0]);
                auto wF32 = floatConst(b, loc, ioTy, 1.0f);
                auto w = castTensor(b, loc, opTy, wF32);
                auto out = mlir::tosa::ArithmeticRightShiftOp::create(
                    b, loc, opTy, lhs, w, b.getBoolAttr(false));
                auto outF32 = castTensor(b, loc, ioTy, out.getResult());
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{outF32});
            });
    });
}

// =============================================================================
// Group 3: Reduction Math Ops
// =============================================================================

TEST_F(ConquerQuantisationTest, AvgPool2DOp) {
    testAllSingleOpConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 4, 4, 3});
        auto outTy = tensorOf(f32(), {1, 2, 2, 3});
        return makeModule({inTy}, {outTy},
            [this, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                auto pool = mlir::tosa::AvgPool2dOp::create(
                    b, loc, outTy,
                    args[0], zpF32(b, loc), zpF32(b, loc),   // input, input_zp, output_zp
                    b.getDenseI64ArrayAttr({2, 2}),            // kernel
                    b.getDenseI64ArrayAttr({2, 2}),            // stride
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}),     // pad
                    mlir::TypeAttr::get(f32()));                // acc_type
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{pool.getResult()});
            });
    });
}

TEST_F(ConquerQuantisationTest, Reduction_reduce_sum)
    { testAllSingleOpConfigs(reductionF32<mlir::tosa::ReduceSumOp>()); }
TEST_F(ConquerQuantisationTest, Reduction_reduce_product)
    { testAllSingleOpConfigs(reductionF32<mlir::tosa::ReduceProductOp>()); }
TEST_F(ConquerQuantisationTest, Reduction_reduce_max)
    { testAllSingleOpConfigs(reductionNanMode<mlir::tosa::ReduceMaxOp>()); }
TEST_F(ConquerQuantisationTest, Reduction_reduce_min)
    { testAllSingleOpConfigs(reductionNanMode<mlir::tosa::ReduceMinOp>()); }
