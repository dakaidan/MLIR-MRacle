#include "conquer/backend/cpu/cpu_target.h"
#include "conquer/backend/iree/iree_target.h"
#include "conquer/core/utils.h"
#include "conquer/core/logging.h"
#include "conquer/loaders/manager.h"
#include "conquer/passes/naming_pass.h"
#include "conquer/passes/analysis/model_proxy.h"
#include "conquer/passes/runner.h"
#include "conquer/quantisation/policy.h"
#include "conquer/quantisation/runner.h"
#include "conquer/runtime/session.h"

#include <llvm/Support/Debug.h>

#include <CLI/CLI.hpp>

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <numeric>


#include "conquer/core/context.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-opt"

using namespace llvm;

/**
 * Configures ConQuER and LLVM debug flags based on CLI arguments.
 * debug_level mapping:
 *   0 = info, 1 = info+debug, 2 = info+debug+trace, 4 = also enable global LLVM_DEBUG output.
 */
void setupLLVMDebug(const bool enable_debug, const std::string &debug_only, const int debug_level) {
    int effective_level = -1;

    if (debug_level >= 0) {
        effective_level = debug_level;
    } else if (enable_debug || !debug_only.empty()) {
        effective_level = 2;
    }

    conquer::set_log_level(effective_level);
    conquer::set_debug_only_filters(debug_only);

    const bool enable_global_llvm_debug = (effective_level >= 4);
    DebugFlag = enable_global_llvm_debug;

    if (enable_global_llvm_debug && !debug_only.empty()) {
        std::vector<std::string> types;
        std::string s;
        std::stringstream ss(debug_only);
        while (std::getline(ss, s, ',')) {
            if (s.empty()) {
                continue;
            }
            types.push_back(s);
            types.push_back(s + "-info");
            types.push_back(s + "-debug");
            types.push_back(s + "-trace");
        }

        std::vector<const char *> c_types;
        c_types.reserve(types.size());
        for (const auto &type : types) {
            c_types.push_back(type.c_str());
        }

        setCurrentDebugTypes(c_types.data(), c_types.size());
    }
}

/**
 * Loads a model from the filesystem using the appropriate loader.
 */
mlir::OwningOpRef<mlir::Operation *> loadModel(const std::string &model_path, mlir::MLIRContext *ctx) {
    L_INFO("Loading model from: " << model_path);
    const auto &loaderManager = conquer::getModelLoaderManager();
    const auto &loader = loaderManager.getLoader(model_path, ctx);
    auto model = loader->loadModel(model_path, ctx);
    L_INFO("Successfully loaded model.");
    return model;
}

/**
 * Loads input data from the filesystem using the appropriate loader(s).
 * Supports multiple comma-separated paths for different inputs.
 */
std::vector<conquer::TensorAllocation> loadData(const std::string &data_path) {
    if (std::filesystem::is_directory(data_path)) {
        std::vector<conquer::TensorAllocation> allocations;
        for (const auto &entry : std::filesystem::directory_iterator(data_path)) {
            if (entry.is_regular_file()) {
                const auto &loaderManager = conquer::getDataLoaderManager();
                const auto &loader = loaderManager.getLoader(entry.path().string());
                auto data = loader->loadData(entry.path().string());
                allocations.insert(allocations.end(), std::make_move_iterator(data.begin()),
                                   std::make_move_iterator(data.end()));
            }
        }
        return allocations;
    }
    std::vector<conquer::TensorAllocation> allocations;
    std::string s;
    std::stringstream ss(data_path);
    while (std::getline(ss, s, ',')) {
        const auto &loaderManager = conquer::getDataLoaderManager();
        const auto &loader = loaderManager.getLoader(s);
        auto data = loader->loadData(s);
        allocations.insert(allocations.end(), std::make_move_iterator(data.begin()),
                           std::make_move_iterator(data.end()));
    }
    return allocations;
}

std::unique_ptr<conquer::Target> createTargetFromName(const std::string &target_name) {
    if (target_name == "cpu") {
        return std::make_unique<conquer::CPUTarget>();
    }
#if IREE_ENABLED
    if (target_name.rfind("iree-", 0) == 0) {
        if (const std::string backend_str = target_name.substr(5); backend_str == "vulkan-spirv") {
            return std::make_unique<conquer::IREEBackendTarget<conquer::IREETargetBackend::VULKAN>>();
        } else if (backend_str == "llvm-cpu") {
            return std::make_unique<conquer::IREEBackendTarget<conquer::IREETargetBackend::CPU>>();
        } else if (backend_str == "adb-cpu") {
            return std::make_unique<conquer::IREEBackendTarget<conquer::IREETargetBackend::ADB_CPU>>();
        } else if (backend_str == "adb-gpu") {
            return std::make_unique<conquer::IREEBackendTarget<conquer::IREETargetBackend::ADB_GPU>>();
        }
#ifdef CONQUER_IREE_HAS_CUDA
        else if (backend_str == "cuda") {
            return std::make_unique<conquer::IREEBackendTarget<conquer::IREETargetBackend::CUDA>>();
        }
#endif
#ifdef CONQUER_IREE_HAS_ROCM
        else if (backend_str == "rocm") {
            return std::make_unique<conquer::IREEBackendTarget<conquer::IREETargetBackend::ROCM>>();
        }
#endif
        else {
            std::cerr << "[Warning] Unknown IREE backend '" << backend_str
                      << "'. Defaulting to 'iree-llvm-cpu' target." << std::endl;
            return std::make_unique<conquer::IREEBackendTarget<conquer::IREETargetBackend::CPU>>();
        }
    }
#endif

    if (!target_name.empty()) {
        std::cerr << "[Warning] Unknown target '" << target_name << "'. Defaulting to 'cpu'." << std::endl;
#if !IREE_ENABLED
        if (target_name.rfind("iree-", 0) == 0) {
            std::cerr << "[Error] IREE targets are not enabled in this build. Please recompile with IREE support "
                         "to use this target."
                      << std::endl;
            std::exit(1);
        }
#endif
    }
    return std::make_unique<conquer::CPUTarget>();
}


/**
 * Command: generate-policy
 * Loads the model, runs the naming pass, and generates a base JSON policy file.
 */
void executeGeneratePolicy(const std::string &model_path, const std::string &output_path) {
    L_INFO("Command: Generate Policy");
    const auto ctx = conquer::createMLIRContext();

    const auto modelOp = loadModel(model_path, ctx.get());

    L_DEBUG("Applying model naming pass.");
    // TODO: Later the naming pass will support grouping based on hardware specifications,
    //       or sensitivity (TE, Hessian, etc. When that is done, we will chose what kind of policy we want to follow
    conquer::runPassOnModule<conquer::ModelNamingPass>(modelOp.get());

    L_INFO("Generating base policy to: " << output_path);

    const auto policy = conquer::generate_default_policy(mlir::cast<mlir::ModuleOp>(modelOp.get()));
    const std::string jsonContent = conquer::to_string(policy);
    std::ofstream outFile(output_path);
    if (!outFile) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }
    outFile << jsonContent;
    outFile.close();

    std::cout << "Base policy generated at: " << output_path << std::endl;
}

/**
 * Command: apply-policy
 * Loads model and config, applies naming, and runs quantization passes.
 * Can optionally save the quantized IR.
 */
void executeApplyPolicy(const std::string &model_path, const std::string &config_path,
                        const std::string &calibration_inputs, const std::string &output_mlir_path) {
    L_INFO("Command: Apply Policy");
    const auto ctx = conquer::createMLIRContext();

    // Load Policy
    L_INFO("Loading configuration from: " << config_path);
    const auto &policy = conquer::parse_policy(conquer::load_file(config_path));

    // Load Model
    const auto modelOp = loadModel(model_path, ctx.get());
    const auto calibrationData = loadData(calibration_inputs);

    for (auto &alloc : calibrationData) {
        L_DEBUG("Calibration Input: " << alloc);
    }

    L_INFO("Applying policy and quantising model.");
    conquer::quantise(modelOp.get(), policy, calibrationData);

    std::string quantisedModelDump;
    raw_string_ostream os(quantisedModelDump);
    modelOp.get()->print(os);
    std::ofstream outFile(output_mlir_path);
    if (!outFile) {
        throw std::runtime_error("Failed to open output file: " + output_mlir_path);
    }
    outFile << os.str();
    outFile.close();

    std::cout << "Policy applied. Model quantised." << std::endl;
}

/**
 * Command: run
 * Loads model, optionally loads a policy, lowers to CPU target, and executes.
 */
void executeRunModel(const std::string &model_path, const std::string &config_path, const std::string &target_name,
                     const std::string &calibration_inputs, const std::string &inputs, const std::string &output) {
    L_INFO("Command: Run Model");
    const auto ctx = conquer::createMLIRContext();

    std::optional<conquer::QuantisationPolicy> policy = std::nullopt;

    // Load Policy
    if (!config_path.empty()) {
        L_INFO("Loading configuration from: " << config_path);
        policy = conquer::parse_policy(conquer::load_file(config_path));
    } else {
        L_INFO("No configuration provided. Running in default precision.");
    }

    // Load Model
    auto modelOp = loadModel(model_path, ctx.get());
    auto inputData = loadData(inputs);
    auto session = conquer::ModelSession();
    session.load(mlir::cast<mlir::ModuleOp>(modelOp.get()));

    if (policy.has_value()) {
        conquer::quantise(modelOp.get(), policy.value(), loadData(calibration_inputs));
    }

    // Lower to Target
    std::unique_ptr<conquer::Target> target = createTargetFromName(target_name);
    L_INFO("Lowering model for CPU target.");

    if (auto err = target->compile(modelOp->clone())) {
        throw std::runtime_error("Failed to lower model for CPU target: " + toString(std::move(err)));
    }

    L_INFO("Successfully lowered model. Executing...");

    auto writeOutput = [&](const std::string &output_str, const std::vector<float> &data) {
        if (output_str.empty()) {
            std::cout << "Output: ";
            for (const auto &val : data) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        } else {
            std::ofstream outFile(output_str);
            if (!outFile) {
                throw std::runtime_error("Failed to open output file: " + output_str);
            }
            for (const auto &val : data) {
                outFile << val << "\n";
            }
            outFile.close();
            std::cout << "Output written to: " << output_str << std::endl;
        }
    };

    for (auto &alloc : inputData) {
        if (!session.validateInputs({alloc.getView()})) {
            throw std::runtime_error("Input tensor does not match model requirements.");
        }
        auto [buffers, views] = session.allocateOutputs(session.resolveOutputShapes({alloc.getView()}));
        if (auto err = target->execute({alloc.getView()}, views)) {
            throw std::runtime_error("Failed to execute: " + toString(std::move(err)));
        }

        for (size_t i = 0; i < views.size(); ++i) {
            auto buffer = buffers.at(i);
            auto view = views.at(i);
            const float *outData = reinterpret_cast<float *>(buffer.data());
            writeOutput(output, std::vector<float>(outData, outData + view.getNumElements()));
        }
    }
}

/**
 * Command: compile
 */
void executeCompileModel(const std::string &model_path, const std::string &config_path, const std::string &target_name,
                     const std::string &calibration_inputs, const std::string &output) {
    L_INFO("Command: Compile Model");
    const auto ctx = conquer::createMLIRContext();

    std::optional<conquer::QuantisationPolicy> policy = std::nullopt;

    // Load Policy
    if (!config_path.empty()) {
        L_INFO("Loading configuration from: " << config_path);
        policy = conquer::parse_policy(conquer::load_file(config_path));
    } else {
        L_INFO("No configuration provided. Running in default precision.");
    }

    // Load Model
    auto modelOp = loadModel(model_path, ctx.get());
    auto session = conquer::ModelSession();
    session.load(mlir::cast<mlir::ModuleOp>(modelOp.get()));

    if (policy.has_value()) {
        conquer::quantise(modelOp.get(), policy.value(), loadData(calibration_inputs));
    }

    // Lower to Target
    std::unique_ptr<conquer::Target> target = createTargetFromName(target_name);
    L_INFO("Lowering model for CPU target.");

    if (auto err = target->compile(modelOp->clone())) {
        throw std::runtime_error("Failed to lower model for CPU target: " + toString(std::move(err)));
    }

    L_INFO("Successfully lowered model. Saving...");

    if (auto err = target->save_compiled_module_to_file(output)) {
        throw std::runtime_error("Failed to save compiled module: " + toString(std::move(err)));
    }
    std::cout << "Compiled module saved to: " << output << std::endl;
}

/**
 * Command: run-precompiled
 */
void executeRunCompiledModel(const std::string &model_path, const std::string &compiled_path,
                     const std::string &target_name, const std::string &inputs, const std::string &output) {
    L_INFO("Command: Run Model");
    const auto ctx = conquer::createMLIRContext();

    // Load Model
    auto modelOp = loadModel(model_path, ctx.get());
    auto inputData = loadData(inputs);
    auto session = conquer::ModelSession();
    session.load(mlir::cast<mlir::ModuleOp>(modelOp.get()));

    // Lower to Target
    std::unique_ptr<conquer::Target> target = createTargetFromName(target_name);
    L_INFO("Lowering model for CPU target.");

    if (auto err = target->load_compiled_module_from_file(compiled_path)) {
        throw std::runtime_error("Failed to load model for target: " + toString(std::move(err)));
    }

    L_INFO("Successfully loaded compiled model. Executing...");

    auto writeOutput = [&](const std::string &output_str, const std::vector<float> &data) {
        if (output_str.empty()) {
            std::cout << "Output: ";
            for (const auto &val : data) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        } else {
            std::ofstream outFile(output_str);
            if (!outFile) {
                throw std::runtime_error("Failed to open output file: " + output_str);
            }
            for (const auto &val : data) {
                outFile << val << "\n";
            }
            outFile.close();
            std::cout << "Output written to: " << output_str << std::endl;
        }
    };

    for (auto &alloc : inputData) {
        if (!session.validateInputs({alloc.getView()})) {
            throw std::runtime_error("Input tensor does not match model requirements.");
        }
        auto [buffers, views] = session.allocateOutputs(session.resolveOutputShapes({alloc.getView()}));
        if (auto err = target->execute({alloc.getView()}, views)) {
            throw std::runtime_error("Failed to execute: " + toString(std::move(err)));
        }

        for (size_t i = 0; i < views.size(); ++i) {
            auto buffer = buffers.at(i);
            auto view = views.at(i);
            const float *outData = reinterpret_cast<float *>(buffer.data());
            writeOutput(output, std::vector<float>(outData, outData + view.getNumElements()));
        }
    }
}

void executeMutateLayer(const std::string &layer_name, const std::string& allowed_precisions_str, const std::string& allowed_calibrations_str, const std::string &output_path) {
    L_INFO("Command: Mutate Layer");

    const auto split_string = [&](const std::string& comma_sep) {
        std::vector<std::string> precs;
        std::string s;
        std::stringstream ss(comma_sep);
        while (std::getline(ss, s, ',')) {
            if (!s.empty()) {
                precs.push_back(s);
            }
        }
        return precs;
    };

    auto supported_precisions = std::vector<conquer::Precision>{};
    for (const auto &prec_str : split_string(allowed_precisions_str)) {
        try {
            supported_precisions.push_back(conquer::from_string(prec_str));
        } catch (const std::exception &e) {
            std::cerr << "[Warning] Invalid precision '" << prec_str << "' in allowed_precisions. Skipping." << std::endl;
        }
    }
    auto allowed_calibrations = std::vector<conquer::CalibrationMethod>{};
    for (const auto &calib_str : split_string(allowed_calibrations_str)) {
        try {
            if (calib_str == "minmax") {
                allowed_calibrations.push_back(conquer::CalibrationMethod::MinMax);
            } else if (calib_str == "percentile") {
                allowed_calibrations.push_back(conquer::CalibrationMethod::Percentile);
            } else if (calib_str == "entropy") {
                allowed_calibrations.push_back(conquer::CalibrationMethod::Entropy);
            } else {
                throw std::invalid_argument("Unknown calibration method: " + calib_str);
            }
        } catch (const std::exception &e) {
            std::cerr << "[Warning] Invalid calibration method '" << calib_str << "' in allowed_calibrations. Skipping." << std::endl;
        }
    }

    L_INFO("Generating mutant policies for layer: " << layer_name);
    const auto mutants = conquer::generate_mutations(layer_name, true, supported_precisions);

    L_INFO("Saving mutant policies to: " << output_path);
    std::ofstream outFile(output_path);
    if (!outFile) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }
    outFile << "[\n";
    for (const auto &mutant : mutants) {
        const std::string jsonContent = conquer::to_string(mutant);

        outFile << jsonContent;
        if (&mutant != &mutants.back()) {
            outFile << ",\n";
        } else {
            outFile << "\n";
        }
    }
    outFile << "]\n";
    outFile.close();

    std::cout << "Mutant policies generated and saved to: " << output_path << std::endl;
}

/**
 * Command: extract-dag
 * Loads the model, runs the naming pass, builds the DAG, and dumps it to JSON.
 */
void executeExtractDAG(const std::string &model_path, const std::string &output_path) {
    L_INFO("Command: Extract DAG");
    const auto ctx = conquer::createMLIRContext();
    const auto modelOp = loadModel(model_path, ctx.get());

    L_DEBUG("Applying model naming pass to ensure nodes are identifiable.");
    conquer::runPassOnModule<conquer::ModelNamingPass>(modelOp.get());

    L_INFO("Building Quantized DAG.");
    auto dag = conquer::build_quantized_dag(mlir::cast<mlir::ModuleOp>(modelOp.get()));

    nlohmann::json j = dag;
    std::ofstream outFile(output_path);
    if (!outFile) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }
    outFile << j.dump(4);
    outFile.close();

    std::cout << "DAG extracted to: " << output_path << std::endl;
}

/**
 * Command: device-info
 * Queries and displays hardware capabilities for a specific target.
 */
void executeDeviceInfo(const std::string &target_name) {
    L_INFO("Command: Device Info");

    conquer::HardwareCapability capability;

    if (target_name == "cpu") {
        capability = conquer::query_cpu_capability();
    }
#if IREE_ENABLED
    else if (target_name == "iree-vulkan-spirv") {
        capability = conquer::query_vulkan_capability();
    } else if (target_name == "iree-llvm-cpu") {
        capability = conquer::query_cpu_capability();
    }
#ifdef CONQUER_IREE_HAS_CUDA
    else if (target_name == "iree-cuda") {
        capability = conquer::query_cuda_capability();
    }
#endif
#ifdef CONQUER_IREE_HAS_ROCM
    else if (target_name == "iree-rocm") {
        capability = conquer::query_rocm_capability();
    }
#endif
    else {
        std::cerr << "[Warning] Unknown target '" << target_name << "'. Defaulting to 'cpu'." << std::endl;
        capability = conquer::query_cpu_capability();
    }
#endif

    // Display Capabilities
    std::cout << capability << std::endl;
}

struct BenchmarkSummary {
    double min_ns = 0.0;
    double max_ns = 0.0;
    double mean_ns = 0.0;
    double median_ns = 0.0;
    double p95_ns = 0.0;
    int warmup = 0;
    int iterations = 0;
    int invocations_per_iteration = 0;
};

static double percentileSorted(const std::vector<double> &sorted, const double q) {
    if (sorted.empty()) return 0.0;
    const double clamped = std::clamp(q, 0.0, 1.0);
    const size_t idx = static_cast<size_t>(clamped * static_cast<double>(sorted.size() - 1));
    return sorted[idx];
}

static BenchmarkSummary summariseSamples(std::vector<double> samples,
                                         const int warmup,
                                         const int iterations,
                                         const int invocations_per_iteration) {
    BenchmarkSummary s;
    s.warmup = warmup;
    s.iterations = iterations;
    s.invocations_per_iteration = invocations_per_iteration;

    if (samples.empty()) {
        return s;
    }

    std::sort(samples.begin(), samples.end());
    s.min_ns = samples.front();
    s.max_ns = samples.back();
    s.mean_ns = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    s.median_ns = percentileSorted(samples, 0.5);
    s.p95_ns = percentileSorted(samples, 0.95);
    return s;
}

void executeBenchmarkCompiledModel(const std::string &model_path,
                                   const std::string &compiled_path,
                                   const std::string &target_name,
                                   const std::string &inputs,
                                   const int warmup,
                                   const int iterations,
                                   const std::string &json_output_path) {
    L_INFO("Command: Benchmark Precompiled Model");
    const auto ctx = conquer::createMLIRContext();

    auto modelOp = loadModel(model_path, ctx.get());
    auto inputData = loadData(inputs);

    if (inputData.empty()) {
        throw std::runtime_error("No input tensors were loaded for benchmarking.");
    }

    auto session = conquer::ModelSession();
    session.load(mlir::cast<mlir::ModuleOp>(modelOp.get()));

    auto target = createTargetFromName(target_name);

    if (auto err = target->load_compiled_module_from_file(compiled_path)) {
        throw std::runtime_error("Failed to load compiled module: " + toString(std::move(err)));
    }

    // Validate all loaded inputs ahead of time.
    for (auto &alloc : inputData) {
        if (!session.validateInputs({alloc.getView()})) {
            throw std::runtime_error("Input tensor does not match model requirements.");
        }
    }

    std::vector<conquer::OutputAllocation> preallocatedOutputs;
    preallocatedOutputs.reserve(inputData.size());

    for (auto &alloc : inputData) {
        auto shapes = session.resolveOutputShapes({alloc.getView()});

        preallocatedOutputs.push_back(session.allocateOutputs(shapes));
    }

    // Warmup.
    for (int w = 0; w < warmup; ++w) {
        for (size_t i = 0; i < inputData.size(); ++i) {
            if (auto err = target->execute({inputData[i].getView()}, preallocatedOutputs[i].views)) {
                throw std::runtime_error("Warmup execution failed: " + toString(std::move(err)));
            }
        }
    }

    std::vector<double> samples_ns;
    samples_ns.reserve(static_cast<size_t>(iterations));

    for (int it = 0; it < iterations; ++it) {
        const auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < inputData.size(); ++i) {
            if (auto err = target->execute({inputData[i].getView()}, preallocatedOutputs[i].views)) {
                throw std::runtime_error("Timed execution failed: " + toString(std::move(err)));
            }
        }

        const auto end = std::chrono::steady_clock::now();
        const double ns =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        samples_ns.push_back(ns);
    }

    const auto summary = summariseSamples(samples_ns, warmup, iterations, static_cast<int>(inputData.size()));

    nlohmann::json j{
        {"warmup", summary.warmup},
        {"iterations", summary.iterations},
        {"invocations_per_iteration", summary.invocations_per_iteration},
        {"min_ns", summary.min_ns},
        {"max_ns", summary.max_ns},
        {"mean_ns", summary.mean_ns},
        {"median_ns", summary.median_ns},
        {"p95_ns", summary.p95_ns}
    };

    if (json_output_path.empty()) {
        std::cout << j.dump(2) << std::endl;
    } else {
        std::ofstream outFile(json_output_path);
        if (!outFile) {
            throw std::runtime_error("Failed to open output file: " + json_output_path);
        }
        outFile << j.dump(2);
        outFile.close();
        std::cout << "Benchmark summary written to: " << json_output_path << std::endl;
    }
}

int main(const int argc, char **argv) {
    CLI::App app{"ConQuER: Configurable Quantisation via Evolutionary Refinement"};

    std::string model_path;

    bool debug_flag = false;
    app.add_flag("--debug", debug_flag, "Enable debug logging.");
    std::string debug_only;
    app.add_option("--debug-only", debug_only, "Enable debug logging for specific tags (comma separated).");
    int debug_level = -1;
    app.add_option("--debug-level", debug_level,
                   "Set logging verbosity (0=info, 1=info+debug, 2=info+debug+trace, 4=also enable global LLVM/MLIR/IREE debug output).")
        ->check(CLI::Range(0, 4));

    app.require_subcommand(1); // Force the user to choose a mode

    // --- Version Information ---
    const auto *cmd_version = app.add_subcommand("version", "Display version information.");

    // --- Generate Policy ---
    // Usage: ./conquer generate-policy -m model -o policy.json
    auto *cmd_gen = app.add_subcommand("generate-policy", "Generate a base quantisation policy JSON from a model.");
    cmd_gen->add_option("-m,--model", model_path, "Path to the model file.")->check(CLI::ExistingFile)->required();
    std::string output_json_path = "policy.json";
    cmd_gen->add_option("-o,--output", output_json_path, "Output path for the generated JSON policy.");

    // --- Apply Policy ---
    // Usage: ./conquer apply-policy -m model -c config.json [--calibration-inputs input1,input2,...] [-o output.mlir]
    auto *cmd_apply =
        app.add_subcommand("apply-policy", "Apply a quantisation policy to a model (Dry run/IR generation).");
    cmd_apply->add_option("-m,--model", model_path, "Path to the model file.")->check(CLI::ExistingFile)->required();
    std::string apply_config_path;
    cmd_apply->add_option("-c,--configuration", apply_config_path, "Path to the JSON configuration file.")
        ->check(CLI::ExistingFile)
        ->required();
    std::string apply_calibration_inputs;
    cmd_apply->add_option("--calibration-inputs", apply_calibration_inputs,
                          "Comma separated list of input files, required for activation calibration");
    std::string output_mlir_path = "quantised_model.mlir";
    cmd_apply->add_option("-o,--output", output_mlir_path, "Output path for the quantised MLIR model.");

    // --- Run Model ---
    // Usage: ./conquer run -m model [-c config.json] [-target cpu|...]
    auto *cmd_run = app.add_subcommand("run", "Load, compile, and execute the model.");
    cmd_run->add_option("-m,--model", model_path, "Path to the model file.")->check(CLI::ExistingFile)->required();
    std::string run_config_path;
    cmd_run->add_option("-c,--configuration", run_config_path, "Optional path to a JSON policy file.");
    std::string execution_mode = "cpu";
    cmd_run->add_option("-t,--target", execution_mode,
                        "Execution target (default: cpu). Currently 'cpu', 'iree-llvm-cpu', 'iree-vulkan-spirv', "
                        "'iree-cuda', and 'iree-rocm' are supported.");
    std::string run_calibration_inputs;
    cmd_run->add_option("--calibration-inputs", run_calibration_inputs,
                        "Comma separated list of input files, required for activation calibration");
    std::string run_input;
    cmd_run->add_option("-i,--input", run_input, "comma separated list of input files, required for execution")
        ->required();
    std::string output_results_path;
    cmd_run->add_option("-o,--output", output_results_path, "Output path for execution results (default: stdout).");

    auto *cmd_compile = app.add_subcommand("compile", "Compile the model for a specific target without executing.");
    cmd_compile->add_option("-m,--model", model_path, "Path to the model file.")->check(CLI::ExistingFile)->required();
    std::string compile_config_path;
    cmd_compile->add_option("-c,--configuration", compile_config_path, "Optional path to a JSON policy file.");
    std::string compile_target = "cpu";
    cmd_compile->add_option("-t,--target", compile_target,
                        "Compilation target (default: cpu). Currently 'cpu', 'iree-llvm-cpu', 'iree-vulkan-spirv', 'iree-cuda', and 'iree-rocm' are supported.");
    std::string compile_calibration_inputs;
    cmd_compile->add_option("--calibration-inputs", compile_calibration_inputs,
                        "Comma separated list of input files, required for activation calibration");
    std::string output_compiled_path = "compiled_module.bin";
    cmd_compile->add_option("-o,--output", output_compiled_path, "Output path for the compiled module.");

    auto *cmd_run_precompiled = app.add_subcommand("run-precompiled", "Run a precompiled module with the original model for reference.");
    cmd_run_precompiled->add_option("-m,--model", model_path, "Path to the model file.")->check(CLI::ExistingFile)->required();
    std::string precompiled_path;
    cmd_run_precompiled->add_option("-c,--compiled", precompiled_path, "Path to the compiled module file.")->check(CLI::ExistingFile)->required();
    std::string precompiled_target = "cpu";
    cmd_run_precompiled->add_option("-t,--target", precompiled_target,
                        "Target of the compiled module (default: cpu). Currently 'cpu', 'iree-llvm-cpu', 'iree-vulkan-spirv', 'iree-cuda', and 'iree-rocm' are supported.");
    std::string precompiled_inputs;
    cmd_run_precompiled->add_option("-i,--input", precompiled_inputs, "comma separated list of input files, required for execution")->required();
    std::string precompiled_output_results_path;
    cmd_run_precompiled->add_option("-o,--output", precompiled_output_results_path, "Output path for execution results (default: stdout).");

    auto *cmd_mutate = app.add_subcommand(
    "mutate-layer",
    "Generate all single-layer mutant policies for a given layer."
    );
    std::string mutate_layer_name;
    cmd_mutate->add_option("-l,--layer", mutate_layer_name,
                           "Name of the layer to mutate.")
        ->required();
    std::string mutate_allowed_precisions;
    cmd_mutate->add_option("-p,--precisions", mutate_allowed_precisions,
                           "Comma separated list of allowed precisions for mutation (e.g. FP32,INT8). If not specified, all supported precisions will be considered.");
    std::string mutate_allowed_calibrations;
    cmd_mutate->add_option("-c,--calibrations", mutate_allowed_calibrations,
                           "Comma separated list of allowed calibration methods for mutation (e.g. minmax, entropy, percentile). If not specified, all supported methods will be considered.");
    std::string mutate_output_path;
    cmd_mutate->add_option("-o,--output", mutate_output_path,
                           "Output path for mutant policies JSON (default: stdout).");

    auto *cmd_extract_dag = app.add_subcommand("extract-dag", "Extracts the model graph topology and memory stats to JSON.");
    cmd_extract_dag->add_option("-m,--model", model_path, "Path to the model file.")->check(CLI::ExistingFile)->required();
    std::string extract_output_path = "dag.json";
    cmd_extract_dag->add_option("-o,--output", extract_output_path, "Output path for the DAG JSON.");

    // --- Device Info ---
    // Usage: ./conquer device-info -target cpu|...
    auto *cmd_device_info =
        app.add_subcommand("device-info", "Query and display hardware capabilities for a specific target.");
    std::string device_info_target = "cpu";
    cmd_device_info->add_option("-t,--target", device_info_target,
                                "Target to query (default: cpu). Currently 'cpu', 'iree-llvm-cpu', "
                                "'iree-vulkan-spirv', 'iree-cuda', and 'iree-rocm' are supported.");

    auto *cmd_benchmark_precompiled = app.add_subcommand("benchmark-precompiled", "Benchmark a precompiled module with warmup/repetitions.");
    cmd_benchmark_precompiled->add_option("-m,--model", model_path, "Path to the original model file.")->check(CLI::ExistingFile)->required();
    std::string benchmark_compiled_path;
    cmd_benchmark_precompiled->add_option("-c,--compiled", benchmark_compiled_path,"Path to the compiled module file.")->check(CLI::ExistingFile)->required();
    std::string benchmark_target = "cpu";
    cmd_benchmark_precompiled->add_option("-t,--target", benchmark_target, "Target of the compiled module.")->required();
    std::string benchmark_inputs;
    cmd_benchmark_precompiled->add_option("-i,--input", benchmark_inputs, "Input file or directory for benchmarking.")->required();
    int benchmark_warmup = 20;
    cmd_benchmark_precompiled->add_option("--warmup", benchmark_warmup, "Number of warmup iterations.");
    int benchmark_iterations = 100;
    cmd_benchmark_precompiled->add_option("--iterations", benchmark_iterations, "Number of timed iterations.");
    std::string benchmark_json_out;
    cmd_benchmark_precompiled->add_option("--json-out", benchmark_json_out,"Optional JSON output path for benchmark summary.");


    // --- Parsing ---
    try {
        CLI11_PARSE(app, argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    // --- Execution ---
    try {
        setupLLVMDebug(debug_flag, debug_only, debug_level);

        if (app.got_subcommand(cmd_gen)) {
            executeGeneratePolicy(model_path, output_json_path);
        } else if (app.got_subcommand(cmd_apply)) {
            executeApplyPolicy(model_path, apply_config_path, apply_calibration_inputs, output_mlir_path);
        } else if (app.got_subcommand(cmd_run)) {
            executeRunModel(model_path, run_config_path, execution_mode, run_calibration_inputs, run_input, output_results_path);
        } else if (app.got_subcommand(cmd_compile)) {
            executeCompileModel(model_path, compile_config_path, compile_target, compile_calibration_inputs, output_compiled_path);
        } else if (app.got_subcommand(cmd_run_precompiled)) {
            executeRunCompiledModel(model_path, precompiled_path, precompiled_target, precompiled_inputs, precompiled_output_results_path);
        } else if (app.got_subcommand(cmd_mutate)) {
            executeMutateLayer(mutate_layer_name, mutate_allowed_precisions, mutate_allowed_calibrations, mutate_output_path);
        } else if (app.got_subcommand(cmd_extract_dag)) {
            executeExtractDAG(model_path, extract_output_path);
        } else if (app.got_subcommand(cmd_device_info)) {
            executeDeviceInfo(device_info_target);
        } else if (app.got_subcommand(cmd_benchmark_precompiled)) {
            executeBenchmarkCompiledModel(model_path, benchmark_compiled_path, benchmark_target,
                                          benchmark_inputs, benchmark_warmup, benchmark_iterations,
                                          benchmark_json_out);
        } else if (app.got_subcommand(cmd_version)) {
            std::cout << "ConQuER Version: 1.0.0" << std::endl;
            std::cout << "LLVM Version: " << LLVM_VERSION_STRING << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}