#include <ranges>
#include <gtest/gtest.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include "conquer/core/context.h"
#include "conquer/core/types.h"
#include "conquer/dialect/ConquerDialect.h"
#include "conquer/quantisation/policy.h"
#include "conquer/quantisation/runner.h"
#include "conquer/passes/naming_pass.h"

using MutationContainer = decltype(conquer::generate_mutations(""));
using MutationType = typename MutationContainer::value_type;

class ConquerQuantisationBlockTest : public ::testing::Test {
protected:
    std::shared_ptr<mlir::MLIRContext> context;

    void SetUp() override {
        context = conquer::createMLIRContext();
        context->getOrLoadDialect<mlir::tosa::TosaDialect>();
        context->getOrLoadDialect<mlir::func::FuncDialect>();
        context->getOrLoadDialect<conquer::ConquerDialect>();
        context->getOrLoadDialect<mlir::arith::ArithDialect>();
    }

    // -------------------------------------------------------------------------
    // Type & Constant Helpers (Reused from Single Op Tests)
    // -------------------------------------------------------------------------
    [[nodiscard]] mlir::FloatType f32() const { return mlir::Float32Type::get(context.get()); }

    static mlir::RankedTensorType tensorOf(const mlir::Type elem, const llvm::ArrayRef<int64_t> shape) {
        return mlir::RankedTensorType::get(shape, elem);
    }

    mlir::Value indexConst(mlir::OpBuilder& b, mlir::Location loc, llvm::ArrayRef<int64_t> values) const {
        auto tensorType = mlir::RankedTensorType::get({static_cast<int64_t>(values.size())}, b.getIndexType());

        std::vector<mlir::Attribute> attrs;
        attrs.reserve(values.size());
        for (int64_t v : values) {
            attrs.push_back(mlir::IntegerAttr::get(b.getIndexType(), v));
        }

        auto attr = mlir::DenseElementsAttr::get(tensorType, attrs);

        // Cast to the explicitly required DenseIntElementsAttr
        auto intAttr = llvm::cast<mlir::DenseIntElementsAttr>(attr);
        mlir::Type shapeType = mlir::tosa::shapeType::get(context.get(), values.size());

        return mlir::tosa::ConstShapeOp::create(b, loc, shapeType, intAttr).getResult();
    }

    static mlir::Value floatConst(mlir::OpBuilder& b, const mlir::Location loc,
                           const mlir::RankedTensorType ty, const float val) {
        int64_t num_elems = ty.getNumElements();
        std::vector<float> values;
        values.reserve(num_elems);

        if (num_elems == 1 || val == 0.0f) {
            for (int64_t i = 0; i < num_elems; ++i) { values.push_back(val); }
        } else {
            for (int64_t i = 0; i < num_elems; ++i) {
                float fraction = static_cast<float>(i) / (num_elems - 1);
                values.push_back(val * (2.0f * fraction - 1.0f));
            }
        }

        const auto attr = mlir::DenseElementsAttr::get(ty, llvm::ArrayRef<float>(values));
        return mlir::tosa::ConstOp::create(b, loc, ty, attr).getResult();
    }

    mlir::Value zpF32(mlir::OpBuilder& b, const mlir::Location loc) const {
        return floatConst(b, loc, tensorOf(f32(), {1}), 0.0f);
    }

    // -------------------------------------------------------------------------
    // Module Factory
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
            for (size_t i = 0; i < numElems; ++i) { ptr[i] = static_cast<float>(i + 1); }
            calibration.push_back(std::move(alloc));
        }
        return calibration;
    }

    // Recursive function to generate the Cartesian product of all mutations across all ops
    void buildPolicyCombinations(
        const std::vector<std::pair<std::string, MutationContainer>>& ops_mutations,
        size_t index,
        conquer::QuantisationPolicy& current_policy,
        std::vector<conquer::QuantisationPolicy>& all_policies) {

        if (index == ops_mutations.size()) {
            all_policies.push_back(current_policy);
            return;
        }

        const auto& [op_name, mutations] = ops_mutations[index];
        for (const auto& mut : mutations) {
            current_policy.layers[op_name] = mut;
            buildPolicyCombinations(ops_mutations, index + 1, current_policy, all_policies);
        }
    }

    // Main test driver for op combinations
    void testAllBlockConfigs(const ModuleFactory& factory, const std::string& blockName) {
        auto probe = factory();
        ASSERT_TRUE(probe) << "Module construction failed for " << blockName;
        conquer::runPassOnModule<conquer::ModelNamingPass>(probe.get());

        auto base_policy = conquer::generate_default_policy(probe.get());
        ASSERT_FALSE(base_policy.layers.empty()) << "No quantisable ops found in " << blockName;

        // Gather all ops and their respective mutations
        std::vector<std::pair<std::string, MutationContainer>> ops_mutations;
        for (const auto &target: base_policy.layers | std::views::keys) {
            auto mutations = conquer::generate_mutations(target);

            std::erase_if(mutations, [](const conquer::Layer& l) {
                if (const auto w_cfg = conquer::get_quant_config(l.weight_policy)) {
                    if (w_cfg->calibration != conquer::CalibrationMethod::MinMax) return true;
                }
                if (const auto a_cfg = conquer::get_quant_config(l.activation_policy)) {
                    if (a_cfg->calibration != conquer::CalibrationMethod::MinMax) return true;
                }
                return false;
            });

            ASSERT_FALSE(mutations.empty()) << "0 mutations for op: " << target;
            ops_mutations.emplace_back(target, mutations);
        }

        // Build all possible combinations of mutations
        std::vector<conquer::QuantisationPolicy> test_policies;
        conquer::QuantisationPolicy temp_policy;
        buildPolicyCombinations(ops_mutations, 0, temp_policy, test_policies);

        llvm::outs() << "[INFO] Testing " << test_policies.size()
                     << " combinatorial policies for block: " << blockName << "\n";
        llvm::outs().flush();

        int success_count = 0;
        for (const auto& policy : test_policies) {
            auto mod = factory();
            ASSERT_TRUE(mod);
            conquer::runPassOnModule<conquer::ModelNamingPass>(mod.get());

            auto mainFunc = mod.get().lookupSymbol<mlir::func::FuncOp>("main");
            auto dummy_calib = buildFp32CalibrationForMain(mainFunc);

            bool is_successful = false;

            EXPECT_NO_THROW({
                conquer::quantise(mod.get(), policy, dummy_calib);

                if (mod.get().verify().succeeded()) {
                    is_successful = true;
                } else {
                    is_successful = false;
                    ADD_FAILURE() << "Verification failed on a combination in " << blockName;
                }
            }) << "Crash inside pipeline during combination testing for " << blockName;

            if (is_successful) {
                ++success_count;
            } else {
                std::string moduleStr;
                llvm::raw_string_ostream rso(moduleStr);
                mod.get()->print(rso);
                llvm::outs() << "=== Failed Combination for " << blockName << " ===\n";
                llvm::outs() << "Policy:\n" << to_string(policy) << "\n";
                llvm::outs() << "Module after quantisation attempt:\n" << rso.str() << "\n";
                llvm::outs() << "Original Module before quantisation:\n";
                auto originalMod = factory();
                std::string originalModuleStr;
                llvm::raw_string_ostream originalRso(originalModuleStr);
                originalMod.get()->print(originalRso);
                llvm::outs() << originalRso.str() << "\n";
                llvm::outs().flush();
            }
        }
        llvm::outs() << "[INFO] Successfully tested " << success_count << " combinations for " << blockName << "\n";
    }
};

TEST_F(ConquerQuantisationBlockTest, Conv2D_ReLU_Block) {
    testAllBlockConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 16, 16, 3});
        auto outTy = tensorOf(f32(), {1, 16, 16, 8});
        return makeModule({inTy}, {outTy},
            [this, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w    = floatConst(b, loc, tensorOf(f32(), {8, 3, 3, 3}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {8}), 1.0f);

                auto conv = mlir::tosa::Conv2DOp::create(
                    b, loc, outTy, args[0], w, bias, zpF32(b, loc), zpF32(b, loc),
                    b.getDenseI64ArrayAttr({1, 1, 1, 1}),
                    b.getDenseI64ArrayAttr({1, 1}),
                    b.getDenseI64ArrayAttr({1, 1}),
                    mlir::TypeAttr::get(f32()), b.getBoolAttr(false));

                auto clamp = mlir::tosa::ClampOp::create(
                b, loc, outTy, conv.getResult(),
                b.getF32FloatAttr(0.0f), b.getF32FloatAttr(6.0f));

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{clamp.getResult()});
            });
    }, "Conv2D_ReLU_Block");
}

TEST_F(ConquerQuantisationBlockTest, MobileNet_Separable_Block) {
    testAllBlockConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 8, 8, 4});
        auto midTy = tensorOf(f32(), {1, 8, 8, 4});
        auto outTy = tensorOf(f32(), {1, 8, 8, 16});
        return makeModule({inTy}, {outTy},
            [this, midTy, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto dw_w    = floatConst(b, loc, tensorOf(f32(), {3, 3, 4, 1}), 1.0f);
                const auto dw_bias = floatConst(b, loc, tensorOf(f32(), {4}), 1.0f);
                auto depthwise = mlir::tosa::DepthwiseConv2DOp::create(
                    b, loc, midTy, args[0], dw_w, dw_bias, zpF32(b, loc), zpF32(b, loc),
                    b.getDenseI64ArrayAttr({1, 1, 1, 1}), b.getDenseI64ArrayAttr({1, 1}),
                    b.getDenseI64ArrayAttr({1, 1}), mlir::TypeAttr::get(f32()), b.getBoolAttr(false));

                const auto pw_w    = floatConst(b, loc, tensorOf(f32(), {16, 1, 1, 4}), 1.0f);
                const auto pw_bias = floatConst(b, loc, tensorOf(f32(), {16}), 1.0f);
                auto pointwise = mlir::tosa::Conv2DOp::create(
                    b, loc, outTy, depthwise.getResult(), pw_w, pw_bias, zpF32(b, loc), zpF32(b, loc),
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}), b.getDenseI64ArrayAttr({1, 1}),
                    b.getDenseI64ArrayAttr({1, 1}), mlir::TypeAttr::get(f32()), b.getBoolAttr(false));

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{pointwise.getResult()});
            });
    }, "MobileNet_Separable_Block");
}

TEST_F(ConquerQuantisationBlockTest, Residual_Add_Block) {
    testAllBlockConfigs([this] {
        auto ty = tensorOf(f32(), {1, 8, 8, 4});
        return makeModule({ty}, {ty},
            [this, ty](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w    = floatConst(b, loc, tensorOf(f32(), {4, 3, 3, 4}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {4}), 1.0f);

                auto conv = mlir::tosa::Conv2DOp::create(
                    b, loc, ty, args[0], w, bias, zpF32(b, loc), zpF32(b, loc),
                    b.getDenseI64ArrayAttr({1, 1, 1, 1}), b.getDenseI64ArrayAttr({1, 1}),
                    b.getDenseI64ArrayAttr({1, 1}), mlir::TypeAttr::get(f32()), b.getBoolAttr(false));

                auto add = mlir::tosa::AddOp::create(b, loc, ty, args[0], conv.getResult());
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{add.getResult()});
            });
    }, "Residual_Add_Block");
}

TEST_F(ConquerQuantisationBlockTest, FFN_Dense_Block) {
    testAllBlockConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 1, 32});
        auto outTy = tensorOf(f32(), {1, 1, 64});
        return makeModule({inTy}, {outTy},
            [this, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w = floatConst(b, loc, tensorOf(f32(), {1, 32, 64}), 1.0f);

                auto mm = mlir::tosa::MatMulOp::create(
                    b, loc, outTy, args[0], w, zpF32(b, loc), zpF32(b, loc));

                const auto bias = floatConst(b, loc, tensorOf(f32(), {1, 1, 64}), 0.5f);
                auto add = mlir::tosa::AddOp::create(b, loc, outTy, mm.getResult(), bias);

                auto clamp = mlir::tosa::ClampOp::create(
                    b, loc, outTy, add.getResult(),
                    b.getF32FloatAttr(0.0f), b.getF32FloatAttr(6.0f));

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{clamp.getResult()});
            });
    }, "FFN_Dense_Block");
}

TEST_F(ConquerQuantisationBlockTest, Conv2D_MaxPool_Block) {
    testAllBlockConfigs([this] {
        auto inTy   = tensorOf(f32(), {1, 16, 16, 3});
        auto convTy = tensorOf(f32(), {1, 16, 16, 8});
        auto poolTy = tensorOf(f32(), {1, 8, 8, 8});
        return makeModule({inTy}, {poolTy},
            [this, convTy, poolTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w    = floatConst(b, loc, tensorOf(f32(), {8, 3, 3, 3}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {8}), 1.0f);

                auto conv = mlir::tosa::Conv2DOp::create(
                    b, loc, convTy, args[0], w, bias, zpF32(b, loc), zpF32(b, loc),
                    b.getDenseI64ArrayAttr({1, 1, 1, 1}), b.getDenseI64ArrayAttr({1, 1}),
                    b.getDenseI64ArrayAttr({1, 1}), mlir::TypeAttr::get(f32()), b.getBoolAttr(false));

                auto pool = mlir::tosa::MaxPool2dOp::create(
                    b, loc, poolTy, conv.getResult(),
                    b.getDenseI64ArrayAttr({2, 2}), b.getDenseI64ArrayAttr({2, 2}),
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}));

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{pool.getResult()});
            });
    }, "Conv2D_MaxPool_Block");
}

TEST_F(ConquerQuantisationBlockTest, Padding_Conv2D_Block) {
    testAllBlockConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 8, 8, 4});
        auto padTy = tensorOf(f32(), {1, 10, 10, 4});
        // FIX: The valid 3x3 conv over a 10x10 padded image results in an 8x8 output
        auto outTy = tensorOf(f32(), {1, 8, 8, 8});
        return makeModule({inTy}, {outTy},
            [this, padTy, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {

                auto padConst = indexConst(b, loc, {0,0, 1,1, 1,1, 0,0});
                auto pad = mlir::tosa::PadOp::create(b, loc, padTy, args[0], padConst);

                const auto w    = floatConst(b, loc, tensorOf(f32(), {8, 3, 3, 4}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {8}), 1.0f);

                auto conv = mlir::tosa::Conv2DOp::create(
                    b, loc, outTy, pad.getResult(), w, bias, zpF32(b, loc), zpF32(b, loc),
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}), b.getDenseI64ArrayAttr({1, 1}),
                    b.getDenseI64ArrayAttr({1, 1}), mlir::TypeAttr::get(f32()), b.getBoolAttr(false));

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{conv.getResult()});
            });
    }, "Padding_Conv2D_Block");
}

TEST_F(ConquerQuantisationBlockTest, Transformer_Head_Block) {
    testAllBlockConfigs([this] {
        auto inTy      = tensorOf(f32(), {1, 16, 64});
        auto reshapeTy = tensorOf(f32(), {1, 16, 4, 16});
        auto transTy   = tensorOf(f32(), {1, 4, 16, 16});
        auto matmulTy  = tensorOf(f32(), {4, 16, 16});
        auto outTy     = tensorOf(f32(), {1, 4, 16, 16});
        return makeModule({inTy}, {outTy},
            [this, reshapeTy, transTy, matmulTy, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {

                auto shapeConst = indexConst(b, loc, {1, 16, 4, 16});
                auto reshape = mlir::tosa::ReshapeOp::create(
                    b, loc, reshapeTy, args[0], shapeConst);

                auto trans = mlir::tosa::TransposeOp::create(
                    b, loc, transTy, reshape.getResult(), b.getDenseI32ArrayAttr({0, 2, 1, 3}));

                auto to3DShapeConst = indexConst(b, loc, {4, 16, 16});
                auto to3DReshape = mlir::tosa::ReshapeOp::create(
                    b, loc, matmulTy, trans.getResult(), to3DShapeConst);

                // Both operands are activations (%4), pass the protected zpF32s
                auto mm = mlir::tosa::MatMulOp::create(
                    b, loc, matmulTy, to3DReshape.getResult(), to3DReshape.getResult(), zpF32(b, loc), zpF32(b, loc));

                auto to4DShapeConst = indexConst(b, loc, {1, 4, 16, 16});
                auto to4DReshape = mlir::tosa::ReshapeOp::create(
                    b, loc, outTy, mm.getResult(), to4DShapeConst);

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{to4DReshape.getResult()});
            });
    }, "Transformer_Head_Block");
}

TEST_F(ConquerQuantisationBlockTest, MatMul_Concat_Block) {
    testAllBlockConfigs([this] {
        auto qTy    = tensorOf(f32(), {1, 16, 32});
        auto wTy    = tensorOf(f32(), {1, 32, 16});
        auto headTy = tensorOf(f32(), {1, 16, 16});
        auto outTy  = tensorOf(f32(), {1, 16, 32});
        return makeModule({qTy}, {outTy},
            [this, wTy, headTy, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                auto w1 = floatConst(b, loc, wTy, 0.5f);
                auto w2 = floatConst(b, loc, wTy, -0.5f);

                auto mm1 = mlir::tosa::MatMulOp::create(b, loc, headTy, args[0], w1, zpF32(b, loc), zpF32(b, loc));
                auto mm2 = mlir::tosa::MatMulOp::create(b, loc, headTy, args[0], w2, zpF32(b, loc), zpF32(b, loc));

                auto concat = mlir::tosa::ConcatOp::create(
                    b, loc, outTy, mlir::ValueRange{mm1.getResult(), mm2.getResult()}, b.getI32IntegerAttr(2));

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{concat.getResult()});
            });
    }, "MatMul_Concat_Block");
}

TEST_F(ConquerQuantisationBlockTest, Reduce_Mean_Block) {
    testAllBlockConfigs([this] {
        auto inTy    = tensorOf(f32(), {1, 16, 16, 4});
        auto interTy = tensorOf(f32(), {1, 1, 16, 4});
        auto outTy   = tensorOf(f32(), {1, 1, 1, 4});
        return makeModule({inTy}, {outTy},
            [outTy, interTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                auto sum1 = mlir::tosa::ReduceSumOp::create(b, loc, interTy, args[0], b.getI32IntegerAttr(1));
                auto sum2 = mlir::tosa::ReduceSumOp::create(b, loc, outTy, sum1.getResult(), b.getI32IntegerAttr(2));

                auto scale = floatConst(b, loc, outTy, 1.0f / 256.0f);

                auto shiftTy = mlir::RankedTensorType::get({1}, b.getI8Type());
                auto shiftAttr = mlir::DenseElementsAttr::get(shiftTy, llvm::ArrayRef<int8_t>({0}));
                auto shiftConst = mlir::tosa::ConstOp::create(b, loc, shiftTy, shiftAttr);

                auto mean  = mlir::tosa::MulOp::create(b, loc, outTy, sum2.getResult(), scale, shiftConst.getResult());

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{mean.getResult()});
            });
    }, "Reduce_Mean_Block");
}

TEST_F(ConquerQuantisationBlockTest, CNN_Output_Head_Block) {
    testAllBlockConfigs([this] {
        auto inTy   = tensorOf(f32(), {1, 8, 8, 16});
        auto convTy = tensorOf(f32(), {1, 8, 8, 32});
        auto poolTy = tensorOf(f32(), {1, 1, 1, 32});
        return makeModule({inTy}, {poolTy},
            [this, convTy, poolTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w    = floatConst(b, loc, tensorOf(f32(), {32, 1, 1, 16}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {32}), 1.0f);

                auto conv = mlir::tosa::Conv2DOp::create(
                    b, loc, convTy, args[0], w, bias, zpF32(b, loc), zpF32(b, loc),
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}), b.getDenseI64ArrayAttr({1, 1}),
                    b.getDenseI64ArrayAttr({1, 1}), mlir::TypeAttr::get(f32()), b.getBoolAttr(false));

                auto pool = mlir::tosa::AvgPool2dOp::create(
                    b, loc, poolTy, conv.getResult(),
                    b.getDenseI64ArrayAttr({8, 8}), b.getDenseI64ArrayAttr({1, 1}),
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}), mlir::TypeAttr::get(f32()));

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{pool.getResult()});
            });
    }, "CNN_Output_Head_Block");
}

TEST_F(ConquerQuantisationBlockTest, TransposeConv2D_Activation_Block) {
    testAllBlockConfigs([this] {
        auto inTy  = tensorOf(f32(), {1, 8, 8, 16});
        auto outTy = tensorOf(f32(), {1, 16, 16, 8});
        return makeModule({inTy}, {outTy},
            [this, outTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                const auto w    = floatConst(b, loc, tensorOf(f32(), {8, 2, 2, 16}), 1.0f);
                const auto bias = floatConst(b, loc, tensorOf(f32(), {8}), 1.0f);

                auto tconv = mlir::tosa::TransposeConv2DOp::create(
                    b, loc, outTy, args[0], w, bias,
                    b.getDenseI64ArrayAttr({0, 0, 0, 0}), b.getDenseI64ArrayAttr({2, 2}),
                    mlir::TypeAttr::get(f32()));

                auto clamp = mlir::tosa::ClampOp::create(
                    b, loc, outTy, tconv.getResult(),
                    b.getF32FloatAttr(0.0f), b.getF32FloatAttr(6.0f));

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{clamp.getResult()});
            });
    }, "TransposeConv2D_Activation_Block");
}

TEST_F(ConquerQuantisationBlockTest, Squeeze_And_Excite_Block) {
    testAllBlockConfigs([this] {
        auto inTy = tensorOf(f32(), {1, 8, 8, 16});
        return makeModule({inTy}, {inTy},
            [inTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                auto sig = mlir::tosa::SigmoidOp::create(b, loc, inTy, args[0]);

                auto shiftTy = mlir::RankedTensorType::get({1}, b.getI8Type());
                auto shiftAttr = mlir::DenseElementsAttr::get(shiftTy, llvm::ArrayRef<int8_t>({0}));
                auto shiftConst = mlir::tosa::ConstOp::create(b, loc, shiftTy, shiftAttr);

                auto mul = mlir::tosa::MulOp::create(b, loc, inTy, args[0], sig.getResult(), shiftConst.getResult());
                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{mul.getResult()});
            });
    }, "Squeeze_And_Excite_Block");
}

TEST_F(ConquerQuantisationBlockTest, HardSwish_Block) {
    testAllBlockConfigs([this] {
        auto inTy = tensorOf(f32(), {1, 8, 8, 4});
        return makeModule({inTy}, {inTy},
            [inTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                auto three = floatConst(b, loc, inTy, 3.0f);
                auto add = mlir::tosa::AddOp::create(b, loc, inTy, args[0], three);

                auto clamp = mlir::tosa::ClampOp::create(
                    b, loc, inTy, add.getResult(), b.getF32FloatAttr(0.0f), b.getF32FloatAttr(6.0f));

                auto shiftTy = mlir::RankedTensorType::get({1}, b.getI8Type());
                auto shiftAttr = mlir::DenseElementsAttr::get(shiftTy, llvm::ArrayRef<int8_t>({0}));
                auto shiftConst = mlir::tosa::ConstOp::create(b, loc, shiftTy, shiftAttr);

                auto mul = mlir::tosa::MulOp::create(b, loc, inTy, args[0], clamp.getResult(), shiftConst.getResult());

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{mul.getResult()});
            });
    }, "HardSwish_Block");
}

TEST_F(ConquerQuantisationBlockTest, LSTM_Cell_State_Block) {
    // Typical pattern: Tosa LSTM inner state math (Mul -> Add -> Tanh)
    testAllBlockConfigs([this] {
        auto stateTy = tensorOf(f32(), {1, 64});

        return makeModule({stateTy}, {stateTy},
            [stateTy](mlir::OpBuilder& b, const mlir::Location loc, const mlir::ValueRange args) {
                auto shiftTy = mlir::RankedTensorType::get({1}, b.getI8Type());
                auto shiftAttr = mlir::DenseElementsAttr::get(shiftTy, llvm::ArrayRef<int8_t>({0}));
                auto shiftConst = mlir::tosa::ConstOp::create(b, loc, shiftTy, shiftAttr);

                // FIX: Use args[0] for both operands to simulate the Activation * Activation
                // multiplication of an LSTM cell state, preserving the single-input test harness.
                auto mul1 = mlir::tosa::MulOp::create(b, loc, stateTy, args[0], args[0], shiftConst.getResult());

                auto input_gate = floatConst(b, loc, stateTy, 0.5f);
                auto new_cand = floatConst(b, loc, stateTy, 0.5f);
                auto mul2 = mlir::tosa::MulOp::create(b, loc, stateTy, input_gate, new_cand, shiftConst.getResult());

                auto add = mlir::tosa::AddOp::create(b, loc, stateTy, mul1.getResult(), mul2.getResult());
                auto tanh = mlir::tosa::TanhOp::create(b, loc, stateTy, add.getResult());

                mlir::func::ReturnOp::create(b, loc, mlir::ValueRange{tanh.getResult()});
            });
    }, "LSTM_Cell_State_Block");
}

#include "conquer/passes/quantisation/integer_patterns/utilities.h"

TEST_F(ConquerQuantisationBlockTest, ChainedCast_AttributeDrop_Repro) {
    mlir::OpBuilder b(context.get());
    auto loc = mlir::UnknownLoc::get(context.get());

    auto f32Ty = tensorOf(f32(), {1, 4, 4, 3});
    auto f16Ty = tensorOf(mlir::Float16Type::get(context.get()), {1, 4, 4, 3});
    auto bf16Ty = tensorOf(mlir::BFloat16Type::get(context.get()), {1, 4, 4, 3});

    // 1. The original compute op
    auto originalOp = floatConst(b, loc, f32Ty, 1.0f);

    // Add a fake calibration stat attribute to simulate a calibrated graph
    originalOp.getDefiningOp()->setAttr("activation_stats.min_max.min", b.getF32FloatAttr(0.0f));

    // 2. Simulate the FloatQuantisationPass inserting chained casts via applyMixedPrecisionCast/ensureType:

    // First cast: ensureType(operand, f16)
    auto cast1 = mlir::tosa::CastOp::create(b, loc, f16Ty, originalOp);
    cast1->setAttr("conquer.cast", b.getUnitAttr()); // Tagged correctly

    auto bridgeCast = mlir::tosa::CastOp::create(b, loc, f32Ty, cast1.getResult());
    bridgeCast->setAttr("conquer.bridge", b.getUnitAttr()); // Matches the fixed compiler!

    // Final cast to bf16
    auto cast2 = mlir::tosa::CastOp::create(b, loc, bf16Ty, bridgeCast.getResult());
    cast2->setAttr("conquer.cast", b.getUnitAttr()); // Tagged correctly

    // 3. Simulate the IntegerQuantisationPass trying to read the original op's stats
    mlir::Value strippedVal = conquer::stripTransientCast(cast2.getResult());
    mlir::Operation* strippedOp = strippedVal.getDefiningOp();

    // If the stripper works perfectly, it will pierce all 3 casts and find originalOp.
    bool foundOriginal = (strippedVal == originalOp);

    EXPECT_TRUE(foundOriginal)
        << "\n[BUG] stripTransientCast got stuck!\n"
        << "Expected to reach: " << originalOp.getDefiningOp()->getName().getStringRef().str() << " (with stats)\n"
        << "But got stuck at:  " << strippedOp->getName().getStringRef().str() << " (without stats)\n\n"
        << "This confirms why getActivationStats() fails during the integer pass.";
}
