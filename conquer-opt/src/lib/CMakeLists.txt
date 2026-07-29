# --- TableGen Includes ---
include_directories(
        ${MLIR_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)

# --- TableGen Generation ---
set(CONQUER_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${CONQUER_GEN_DIR})

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/conquer/dialect/ConquerOps.td)
mlir_tablegen(generated/ConquerOps.h.inc -gen-op-decls)
mlir_tablegen(generated/ConquerOps.cpp.inc -gen-op-defs)
mlir_tablegen(generated/ConquerDialect.h.inc -gen-dialect-decls -dialect=conquer)
mlir_tablegen(generated/ConquerDialect.cpp.inc -gen-dialect-defs -dialect=conquer)

add_public_tablegen_target(ConquerOpsIncGen)

# --- Library Target ---
file(GLOB_RECURSE CONQUER_LIB_SOURCES CONFIGURE_DEPENDS *.cpp)

add_library(conquer_lib STATIC ${CONQUER_LIB_SOURCES})
target_compile_options(conquer_lib PRIVATE -fno-rtti)
add_dependencies(conquer_lib ConquerOpsIncGen)

# --- Includes ---
target_include_directories(conquer_lib PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CONQUER_GEN_DIR}
)

target_include_directories(conquer_lib SYSTEM PUBLIC
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${openga_SOURCE_DIR}/src
)

# --- Link Libraries ---
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

target_link_libraries(conquer_lib PUBLIC
        nlohmann_json::nlohmann_json
        CLI11::CLI11
        Eigen3::Eigen

        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRPass
        MLIRTransforms
        mlir_c_runner_utils
        mlir_runner_utils
        MLIRExecutionEngine
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation

        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        ${mlir_translation_libs}

        ${llvm_libs}
        ${amd_libs}
        ${nvptx_libs}
)

register_iree_compile_dependency(conquer_lib)
iree_compile_definitions(conquer_lib)
iree_include_directories(conquer_lib)
add_iree_link_libraries(conquer_lib)
add_mlir_registration_helpers(conquer_lib)

try_make_cuda_available(conquer_lib)
try_make_rocm_available(conquer_lib)
try_make_vulkan_available(conquer_lib)