#include "conquer/backend/capabilities.h"
#include "conquer/core/logging.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#ifdef CONQUER_HAS_CUDA
#include "nvml.h"
#endif
#ifdef CONQUER_HAS_ROCM
#include "rocm_smi/rocm_smi.h"
#include <hip/hip_runtime.h>
#endif
#ifdef CONQUER_HAS_VULKAN
#include <vulkan/vulkan.h>
#ifndef VK_COMPONENT_TYPE_BFLOAT16_KHR
#define VK_COMPONENT_TYPE_BFLOAT16_KHR static_cast<VkComponentTypeKHR>(11)
#endif
#ifndef VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT
#define VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT static_cast<VkComponentTypeKHR>(1000497000)
#endif
#ifndef VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT
#define VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT static_cast<VkComponentTypeKHR>(1000497001)
#endif
#endif

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-capability"

static conquer::DTypeCapability cap(const conquer::DataType dt, const conquer::DTypeSupport s) { return {dt, s}; }

static conquer::DTypeCapability native(const conquer::DataType dt) { return cap(dt, conquer::DTypeSupport::Native); }
static conquer::DTypeCapability emulated(const conquer::DataType dt) {
    return cap(dt, conquer::DTypeSupport::Emulated);
}
static conquer::DTypeCapability unsupported(const conquer::DataType dt) {
    return cap(dt, conquer::DTypeSupport::Unsupported);
}

conquer::HardwareCapability conquer::query_cpu_capability() {
    const llvm::StringMap<bool> features = llvm::sys::getHostCPUFeatures();

    auto has = [&](const std::initializer_list<llvm::StringRef> flags) {
        for (const auto f : flags)
            if (features.lookup(f))
                return true;
        return false;
    };

    const bool avx2 = has({"avx2"});
    const bool avx512f = has({"avx512f"});
    const bool avx512_int = has({"avx512bw", "avx512dq"});

    const bool vnni = has({"avx512vnni", "avxvnni", "avxvnniint8"});
    const bool bf16_hw = has({"avx512bf16", "amx-bf16"});
    const bool fp16_hw = has({"avx512fp16", "amx-fp16"});
    const bool amx_i8 = has({"amx-int8"});

    HardwareCapability cap;
    cap.device_name = llvm::sys::getHostCPUName().str();
    cap.device_uri = "";
    cap.dtypes = {
        native(DataType::FP32),
        fp16_hw ? native(DataType::FP16) : (avx2 ? emulated(DataType::FP16) : unsupported(DataType::FP16)),
        bf16_hw ? native(DataType::BF16) : (avx512f ? emulated(DataType::BF16) : unsupported(DataType::BF16)),
        unsupported(DataType::FP8_E4M3), // TODO: soon to be added it seems
        unsupported(DataType::FP8_E5M2), // TODO: soon to be added it seems
        native(DataType::INT32),
        (avx2 || avx512_int) ? native(DataType::INT16) : emulated(DataType::INT16),
        (vnni || amx_i8 || avx512_int) ? native(DataType::INT8)
                                       : (avx2 ? emulated(DataType::INT8) : unsupported(DataType::INT8)),
        // (vnni || amx_i8 || avx512_int) ? native(DataType::UINT8)
        //                                : (avx2 ? emulated(DataType::UINT8) : unsupported(DataType::UINT8)),
        avx2 ? emulated(DataType::INT4) : unsupported(DataType::INT4),
        // avx2 ? emulated(DataType::UINT4) : unsupported(DataType::UINT4),
    };
    return cap;
}
conquer::HardwareCapability conquer::query_cuda_capability() {
#if defined(CONQUER_HAS_CUDA)
    int sm_major = 0, sm_minor = 0;
    std::string device_name = "Unknown CUDA Device";
    int target_device_index = 0;

    L_DEBUG("Initialising NVML for CUDA query...");
    if (nvmlInit_v2() == NVML_SUCCESS) {
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex_v2(0, &device) == NVML_SUCCESS) {
            nvmlDeviceGetCudaComputeCapability(device, &sm_major, &sm_minor);
            char name[NVML_DEVICE_NAME_BUFFER_SIZE];
            if (nvmlDeviceGetName(device, name, sizeof(name)) == NVML_SUCCESS) {
                device_name = name;
            }
            L_INFO("Found CUDA Device: " << device_name << " (SM " << sm_major << "." << sm_minor << ")");
        } else {
            L_DEBUG("Failed to get NVML device handle.");
        }
        nvmlShutdown();
    } else {
        L_DEBUG("Failed to initialise NVML.");
    }
    if (sm_major == 0)
        return HardwareCapability{};

    auto sm = [&](const int maj, const int min) { return sm_major > maj || (sm_major == maj && sm_minor >= min); };

    const bool has_fp16 = sm(5, 3);
    const bool has_bf16 = sm(8, 0);
    const bool has_fp8 = sm(8, 9);
    const bool has_int8 = sm(6, 1);
    const bool has_hw_int4 = sm(7, 5) && !sm(9, 0);

    HardwareCapability cap;
    cap.device_name = device_name;
    cap.device_uri = "cuda://" + std::to_string(target_device_index);
    cap.dtypes = {
        native(DataType::FP32),
        has_fp16 ? native(DataType::FP16) : unsupported(DataType::FP16),
        has_bf16 ? native(DataType::BF16) : (has_fp16 ? emulated(DataType::BF16) : unsupported(DataType::BF16)),
        has_fp8 ? native(DataType::FP8_E4M3) : unsupported(DataType::FP8_E4M3),
        has_fp8 ? native(DataType::FP8_E5M2) : unsupported(DataType::FP8_E5M2),
        native(DataType::INT32),
        emulated(DataType::INT16),
        has_int8 ? native(DataType::INT8) : unsupported(DataType::INT8),
        has_hw_int4 ? native(DataType::INT4) : (has_int8 ? emulated(DataType::INT4) : unsupported(DataType::INT4)),
    };
    return cap;
#else
    L_DEBUG("CUDA capability query requested but CONQUER_HAS_CUDA not set.");
    return HardwareCapability{};
#endif
}

conquer::HardwareCapability conquer::query_rocm_capability() {
#if defined(CONQUER_HAS_ROCM)
    std::string arch_name = "";
    std::string device_name = "Unknown ROCm Device";
    int target_device_index = 0;

    L_DEBUG("Querying HIP/ROCm device properties...");
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
        arch_name = props.gcnArchName;
        device_name = props.name;
        L_INFO("Found ROCm Device: " << device_name << " (Arch: " << arch_name << ")");
    } else {
        L_DEBUG("Failed to get HIP device properties.");
        return HardwareCapability{};
    }
    auto starts_with = [&](const std::string &prefix) { return arch_name.rfind(prefix, 0) == 0; };

    const bool is_cdna1_plus = starts_with("gfx908") || starts_with("gfx90a") || starts_with("gfx94");
    const bool is_cdna3_plus = starts_with("gfx94");
    const bool is_rdna3 = starts_with("gfx11");

    constexpr bool fp16_hw = true;
    const bool bf16_hw = is_cdna1_plus || is_rdna3;
    const bool fp8_hw = is_cdna3_plus;
    const bool int8_hw = is_cdna1_plus || is_rdna3 || starts_with("gfx10");
    constexpr bool int4_hw = false;

    HardwareCapability cap;
    cap.device_name = device_name.empty() ? arch_name : device_name;
    cap.device_uri = "rocm://" + std::to_string(target_device_index);
    cap.dtypes = {
        native(DataType::FP32),
        fp16_hw ? native(DataType::FP16) : emulated(DataType::FP16),
        bf16_hw ? native(DataType::BF16) : unsupported(DataType::BF16),
        fp8_hw ? native(DataType::FP8_E4M3) : unsupported(DataType::FP8_E4M3),
        fp8_hw ? native(DataType::FP8_E5M2) : unsupported(DataType::FP8_E5M2),
        native(DataType::INT32),
        emulated(DataType::INT16),
        int8_hw ? native(DataType::INT8) : emulated(DataType::INT8),
        int4_hw ? native(DataType::INT4) : (int8_hw ? emulated(DataType::INT4) : unsupported(DataType::INT4)),
    };
    return cap;
#else
    L_DEBUG("ROCM capability query requested but CONQUER_HAS_ROCM not set.");
    return HardwareCapability{};
#endif
}

conquer::HardwareCapability conquer::query_vulkan_capability() {
#if defined(CONQUER_HAS_VULKAN)
    L_DEBUG("Initialising Vulkan capability query...");

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo inst_ci{};
    inst_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_ci.pApplicationInfo = &app_info;

    VkInstance vk_instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&inst_ci, nullptr, &vk_instance) != VK_SUCCESS) {
        L_DEBUG("Failed to create Vulkan instance.");
        return HardwareCapability{};
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(vk_instance, &device_count, nullptr);
    if (device_count == 0) {
        L_DEBUG("No Vulkan physical devices found.");
        vkDestroyInstance(vk_instance, nullptr);
        return HardwareCapability{};
    }

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    vkEnumeratePhysicalDevices(vk_instance, &device_count, physical_devices.data());

    uint32_t selected_device_index = 0;
    VkPhysicalDevice pd = physical_devices[0];
    for (uint32_t i = 0; i < physical_devices.size(); ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(physical_devices[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            pd = physical_devices[i];
            selected_device_index = i;
            break;
        }
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    L_INFO("Found Vulkan Device: " << props.deviceName);
    std::string device_name = props.deviceName;

    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> available_exts(ext_count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, available_exts.data());

    auto has_ext = [&](const char *ext_name) {
        for (const auto &[extensionName, specVersion] : available_exts) {
            if (std::string_view(extensionName) == ext_name)
                return true;
        }
        return false;
    };

    const bool has_float8_ext = has_ext("VK_EXT_shader_float8");
    const bool has_coop_mat_ext = has_ext("VK_KHR_cooperative_matrix");

    VkPhysicalDeviceShaderFloat16Int8Features float16_int8{};
    float16_int8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

    VkPhysicalDevice16BitStorageFeatures storage16{};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

    VkPhysicalDevice8BitStorageFeatures storage8{};
    storage8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;

    void *pNext_head = &float16_int8;
    float16_int8.pNext = &storage16;
    storage16.pNext = &storage8;

    struct VkPhysicalDeviceShaderFloat8FeaturesEXT_Shadow {
        VkStructureType sType;
        void *pNext;
        VkBool32 shaderFloat8;
    } float8_features{};

    if (has_float8_ext) {
        float8_features.sType = static_cast<VkStructureType>(1000497000);
        float8_features.pNext = pNext_head;
        pNext_head = &float8_features;
    }

    struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR_Shadow {
        VkStructureType sType;
        void *pNext;
        VkBool32 cooperativeMatrix;
        VkBool32 cooperativeMatrixRobustBufferAccess;
    } coop_features{};

    if (has_coop_mat_ext) {
        coop_features.sType = static_cast<VkStructureType>(1000506000);
        coop_features.pNext = pNext_head;
        pNext_head = &coop_features;
    }

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = pNext_head;

    vkGetPhysicalDeviceFeatures2(pd, &features2);

    const bool shader_float16 = float16_int8.shaderFloat16;
    const bool shader_int8 = float16_int8.shaderInt8;
    const bool shader_int16 = features2.features.shaderInt16;
    const bool storage_16bit = storage16.storageBuffer16BitAccess;
    const bool storage_8bit = storage8.storageBuffer8BitAccess;
    const bool shader_float8 = has_float8_ext ? float8_features.shaderFloat8 : false;

    bool coop_matrix_fp16 = false;
    bool coop_matrix_bf16 = false;
    bool coop_matrix_fp8 = false;
    bool coop_matrix_int8 = false;
    bool coop_matrix_int4 = false;

    if (has_coop_mat_ext && coop_features.cooperativeMatrix) {
        auto vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
            reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
                vkGetInstanceProcAddr(vk_instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

        if (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
            uint32_t prop_count = 0;
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &prop_count, nullptr);
            std::vector<VkCooperativeMatrixPropertiesKHR> coop_props(prop_count);
            for (auto &p : coop_props)
                p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;

            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &prop_count, coop_props.data());

            for (const auto &prop : coop_props) {
                switch (prop.AType) {
                case VK_COMPONENT_TYPE_FLOAT16_KHR:
                    coop_matrix_fp16 = true;
                    break;
                case VK_COMPONENT_TYPE_BFLOAT16_KHR:
                    coop_matrix_bf16 = true;
                    break;
                case VK_COMPONENT_TYPE_SINT8_KHR:
                case VK_COMPONENT_TYPE_UINT8_KHR:
                    coop_matrix_int8 = true;
                    break;
                case VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT:
                case VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT:
                    coop_matrix_fp8 = true;
                    break;
                default:
                    break;
                }
            }
        }
    }

    vkDestroyInstance(vk_instance, nullptr);

    const bool true_fp16 = shader_float16 && storage_16bit;
    const bool true_int16 = shader_int16 && storage_16bit;
    const bool true_int8 = shader_int8 && storage_8bit;
    const bool true_fp8 = shader_float8 && storage_8bit;

    HardwareCapability cap;
    cap.device_name = device_name;
    cap.device_uri = "vulkan://" + std::to_string(selected_device_index);
    cap.dtypes = {
        native(DataType::FP32),
        (true_fp16 || coop_matrix_fp16) ? native(DataType::FP16) : emulated(DataType::FP16),
        coop_matrix_bf16 ? native(DataType::BF16) : unsupported(DataType::BF16),
        (true_fp8 || coop_matrix_fp8) ? native(DataType::FP8_E4M3) : unsupported(DataType::FP8_E4M3),
        (true_fp8 || coop_matrix_fp8) ? native(DataType::FP8_E5M2) : unsupported(DataType::FP8_E5M2),

        native(DataType::INT32),
        true_int16 ? native(DataType::INT16) : emulated(DataType::INT16),

        (true_int8 || coop_matrix_int8) ? native(DataType::INT8) : emulated(DataType::INT8),

        coop_matrix_int4 ? native(DataType::INT4)
                         : (true_int8 ? emulated(DataType::INT4) : unsupported(DataType::INT4)),
    };
    return cap;

#else
    L_DEBUG("Vulkan capability query requested but CONQUER_HAS_VULKAN not set.");
    return HardwareCapability{};
#endif
}