# --- Options ---
option(CONQUER_ENABLE_IREE "Enable IREE integration and use its bundled LLVM/MLIR" ON)
option(CONQUER_USE_SYSTEM_LLVM "Use system installed LLVM/MLIR (Only applies if IREE is OFF)" ON)

# --- Hardware Backend Options ---
option(CONQUER_BACKEND_CUDA "Build with CUDA support (Requires local CUDA Toolkit)" OFF)
option(CONQUER_BACKEND_ROCM "Build with ROCm support (Requires local ROCm SDK)" OFF)
option(CONQUER_BACKEND_VULKAN "Build with Vulkan support (Requires local Vulkan SDK)" OFF)

# ==========================================
#      Helper Functions
# ==========================================

function(register_iree_compile_dependency target)
    if (CONQUER_ENABLE_IREE)
        add_dependencies(${target} iree-compile)
    endif()
endfunction()

function(iree_compile_definitions target)
    if (CONQUER_ENABLE_IREE)
        set(IREE_COMPILER_SO_PATH "${PROJECT_BINARY_DIR}/third_party/iree/lib/libIREECompiler.so")
        target_compile_definitions(${target} PUBLIC
                IREE_ENABLED=1
                IREE_COMPILER_SO_PATH="${IREE_COMPILER_SO_PATH}"
        )
    else()
        target_compile_definitions(${target} PUBLIC IREE_ENABLED=0)
    endif()
endfunction()

function(iree_include_directories target)
    if (CONQUER_ENABLE_IREE)
        target_include_directories(${target} SYSTEM PUBLIC
                "${PROJECT_SOURCE_DIR}/third_party/iree/compiler/bindings/c"
                "${PROJECT_SOURCE_DIR}/third_party/iree/compiler/src"
                "${PROJECT_SOURCE_DIR}/third_party/iree/runtime/src"
                "${PROJECT_SOURCE_DIR}/third_party/iree/llvm-external-projects/iree-dialects/include"
                "${PROJECT_BINARY_DIR}/third_party/iree/compiler/src"
                "${PROJECT_BINARY_DIR}/third_party/iree/llvm-external-projects/iree-dialects/include"
        )
    endif()
endfunction()

function(add_iree_link_libraries target)
    if (CONQUER_ENABLE_IREE)
        target_link_libraries(${target} PRIVATE
                "${PROJECT_BINARY_DIR}/third_party/iree/lib/libIREECompiler.so"
                iree_runtime_runtime
                iree_compiler_Tools_iree_compile_lib
        )
    endif()
endfunction()

function(conditionally_add_sdk target compile_defs sdk_includes sdk_libs)
    target_compile_definitions(${target} PUBLIC ${compile_defs})

    get_target_property(existing_includes ${target} INCLUDE_DIRECTORIES)
    if(NOT existing_includes)
        set(existing_includes "")
    endif()
    if(sdk_includes)
        foreach(inc IN LISTS sdk_includes)
            if(NOT "${inc}" IN_LIST existing_includes)
                target_include_directories(${target} SYSTEM PUBLIC "${inc}")
            endif()
        endforeach()
    endif()

    get_target_property(existing_libs ${target} LINK_LIBRARIES)
    if(NOT existing_libs)
        set(existing_libs "")
    endif()
    if(sdk_libs)
        foreach(lib IN LISTS sdk_libs)
            if(NOT "${lib}" IN_LIST existing_libs)
                target_link_libraries(${target} PUBLIC "${lib}")
            endif()
        endforeach()
    endif()
endfunction()

function(try_make_cuda_available target)
    if(CONQUER_BACKEND_CUDA)
        message(STATUS "CUDA Backend Enabled: Searching for local CUDA SDK...")
        find_package(CUDAToolkit QUIET)

        set(CUDA_DEFS "CONQUER_HAS_CUDA;CONQUER_IREE_HAS_CUDA")

        if(CUDAToolkit_FOUND)
            conditionally_add_sdk(${target} "${CUDA_DEFS}" "${CUDAToolkit_INCLUDE_DIRS}" "CUDA::cudart;CUDA::nvml")
        else()
            find_package(CUDA REQUIRED)
            conditionally_add_sdk(${target} "${CUDA_DEFS}" "${CUDA_INCLUDE_DIRS}" "${CUDA_LIBRARIES};nvidia-ml")
        endif()
    endif()
endfunction()

function(try_make_rocm_available target)
    if(CONQUER_BACKEND_ROCM)
        message(STATUS "ROCm Backend Enabled: Searching for ROCm SDK...")
        find_package(hip QUIET)

        set(ROCM_DEFS "CONQUER_HAS_ROCM;CONQUER_IREE_HAS_ROCM")

        if(hip_FOUND)
            conditionally_add_sdk(${target} "${ROCM_DEFS}" "${HIP_INCLUDE_DIRS}" "hip::host")
        else()
            find_package(ROCM REQUIRED)
            conditionally_add_sdk(${target} "${ROCM_DEFS}" "${ROCM_INCLUDE_DIRS}" "${ROCM_LIBRARIES}")
        endif()
    endif()
endfunction()

function(try_make_vulkan_available target)
    if(CONQUER_BACKEND_VULKAN)
        message(STATUS "Vulkan Backend Enabled: Searching for Vulkan SDK...")
        find_package(Vulkan REQUIRED)

        set(VK_DEFS "CONQUER_HAS_VULKAN;CONQUER_IREE_HAS_VULKAN")

        conditionally_add_sdk(${target} "${VK_DEFS}" "${Vulkan_INCLUDE_DIRS}" "${Vulkan_LIBRARIES}")
    endif()
endfunction()

function(add_mlir_registration_helpers target)
    if(TARGET MLIRInitAllDialects)
        target_link_libraries(${target} PRIVATE MLIRInitAllDialects)
    elseif(TARGET MLIRRegisterAllDialects)
        target_link_libraries(${target} PRIVATE MLIRRegisterAllDialects)
    else()
        message(FATAL_ERROR "Could not find MLIR helper target for registerAllDialects")
    endif()

    if(TARGET MLIRInitAllExtensions)
        target_link_libraries(${target} PRIVATE MLIRInitAllExtensions)
    elseif(TARGET MLIRRegisterAllExtensions)
        target_link_libraries(${target} PRIVATE MLIRRegisterAllExtensions)
    else()
        message(FATAL_ERROR "Could not find MLIR helper target for registerAllExtensions")
    endif()
endfunction()

# ==========================================
#      Master Setup Macro
# ==========================================

macro(setup_conquer_llvm_and_iree)
    if(CONQUER_ENABLE_IREE)
        message(STATUS "Configuring ConQuER with IREE integration...")

        set(IREE_PATCH_FILE "${PROJECT_BINARY_DIR}/iree_pr_24121.patch")
        if(NOT EXISTS "${IREE_PATCH_FILE}")
            message(STATUS "Downloading IREE PR #24121 patch...")
            file(DOWNLOAD "https://github.com/iree-org/iree/pull/24121.patch" "${IREE_PATCH_FILE}")

            message(STATUS "Applying IREE PR #24121 patch to third_party/iree...")
            execute_process(
                    COMMAND patch -p1 --forward --batch -i "${IREE_PATCH_FILE}"
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/third_party/iree"
                    RESULT_VARIABLE patch_res
            )
            if(patch_res EQUAL 0)
                message(STATUS "Successfully patched IREE with PR #24121.")
            else()
                message(STATUS "Patch was already applied or could not be applied. Continuing...")
            endif()
        endif()

        set(IREE_BUILD_COMPILER ON)
        set(IREE_BUILD_TESTS OFF)
        set(IREE_BUILD_SAMPLES OFF)
        add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)

        set(IREE_LLVM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/iree/third_party/llvm-project")
        set(IREE_LLVM_BIN_DIR "${PROJECT_BINARY_DIR}/third_party/iree/llvm-project")

        list(APPEND CMAKE_MODULE_PATH "${IREE_LLVM_SRC_DIR}/llvm/cmake/modules")
        list(APPEND CMAKE_MODULE_PATH "${IREE_LLVM_SRC_DIR}/mlir/cmake/modules")

        set(LLVM_INCLUDE_DIRS
                "${IREE_LLVM_SRC_DIR}/llvm/include"
                "${IREE_LLVM_BIN_DIR}/include"
        )

        set(MLIR_INCLUDE_DIRS
                "${IREE_LLVM_SRC_DIR}/mlir/include"
                "${IREE_LLVM_BIN_DIR}/tools/mlir/include"
        )

        add_definitions(-DLLVM_ENABLE_DUMP)

    else()
        if(CONQUER_USE_SYSTEM_LLVM)
            message(STATUS "IREE Disabled. Searching for System LLVM/MLIR...")

            find_package(MLIR REQUIRED CONFIG)
            find_package(LLVM REQUIRED CONFIG)

            message(STATUS "Found MLIR Version: ${MLIR_VERSION}")
            message(STATUS "Using MLIR from: ${MLIR_DIR}")

            list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
            list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

            include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
            add_definitions(${LLVM_DEFINITIONS})
        else()
            message(STATUS "IREE Disabled. Building LLVM/MLIR from source (This will take a long time)...")

            set(CONQUER_LLVM_TARGETS "Native" CACHE STRING "Semicolon-separated list of LLVM targets to build")
            set(CONQUER_LLVM_VERSION "2078da43e25a4623cab2d0d60decddf709aaea28" CACHE STRING "Git tag or commit hash for LLVM source")

            set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
            set(LLVM_TARGETS_TO_BUILD "${CONQUER_LLVM_TARGETS}" CACHE STRING "" FORCE)
            set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
            set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
            set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
            set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)

            set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
            set(LLVM_ENABLE_EH ON CACHE BOOL "" FORCE)
            set(LLVM_OPTIMIZED_TABLEGEN ON CACHE BOOL "" FORCE)

            FetchContent_Declare(
                    llvm_project
                    GIT_REPOSITORY https://github.com/llvm/llvm-project.git
                    GIT_TAG        ${CONQUER_LLVM_VERSION}
                    SOURCE_SUBDIR  llvm
            )
            FetchContent_MakeAvailable(llvm_project)

            list(APPEND CMAKE_MODULE_PATH "${llvm_project_SOURCE_DIR}/llvm/cmake/modules")
            list(APPEND CMAKE_MODULE_PATH "${llvm_project_SOURCE_DIR}/mlir/cmake/modules")

            set(LLVM_INCLUDE_DIRS
                    "${llvm_project_SOURCE_DIR}/llvm/include"
                    "${llvm_project_BINARY_DIR}/include"
            )

            set(MLIR_INCLUDE_DIRS
                    "${llvm_project_SOURCE_DIR}/mlir/include"
                    "${llvm_project_BINARY_DIR}/tools/mlir/include"
            )

            add_definitions(-DLLVM_ENABLE_DUMP)
        endif()
    endif()

    # --- Common LLVM/MLIR Setup ---
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)

    llvm_map_components_to_libnames(llvm_libs
            native
            X86AsmParser
            Core
            Support
            ExecutionEngine
            OrcJIT
            MC
            Target
    )

    # AMDGPU and NVPTX targets for standalone LLVM source build
    if(NOT CONQUER_ENABLE_IREE AND NOT CONQUER_USE_SYSTEM_LLVM)
        string(FIND "${CONQUER_LLVM_TARGETS}" "AMDGPU" _has_amdgpu)
        if(NOT _has_amdgpu EQUAL -1)
            llvm_map_components_to_libnames(amd_libs AMDGPUCodeGen AMDGPUDesc AMDGPUInfo)
            list(APPEND llvm_libs ${amd_libs})
        endif()

        string(FIND "${CONQUER_LLVM_TARGETS}" "NVPTX" _has_nvptx)
        if(NOT _has_nvptx EQUAL -1)
            llvm_map_components_to_libnames(nvptx_libs NVPTXCodeGen NVPTXDesc NVPTXInfo)
            list(APPEND llvm_libs ${nvptx_libs})
        endif()
    endif()
endmacro()
