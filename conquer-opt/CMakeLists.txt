cmake_minimum_required(VERSION 3.20)
project(ConQuER DESCRIPTION "Configurable Quantisation via Evolutionary Refinement" LANGUAGES CXX C)

cmake_policy(VERSION 3.10...3.20)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Keep dependencies optimized and portable by default (no -march=native).
set(CONQUER_PROJECT_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
set(CONQUER_DEPS_BUILD_TYPE "Release" CACHE STRING
        "Build type used for LLVM/IREE/third-party dependencies")
set_property(CACHE CONQUER_DEPS_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
option(CONQUER_DEPS_FOLLOW_PROJECT_BUILD_TYPE
        "Build dependencies with the same build type as ConQuER targets"
        OFF)

# Add our custom cmake module directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 1. Fetch external third-party dependencies (JSON, CLI11)
if(NOT CONQUER_DEPS_FOLLOW_PROJECT_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "${CONQUER_DEPS_BUILD_TYPE}")
    message(STATUS "ConQuER dependencies build type: ${CMAKE_BUILD_TYPE}")
endif()

include(Dependencies)

# 2. Include our LLVM/IREE setup scripts
include(SetupLLVM)

# 3. Execute the single, unified master setup macro
setup_conquer_llvm_and_iree()

if(NOT CONQUER_DEPS_FOLLOW_PROJECT_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "${CONQUER_PROJECT_BUILD_TYPE}")
    message(STATUS "ConQuER project build type: ${CMAKE_BUILD_TYPE}")
endif()

# 4. Build the core library (src/lib)
add_subdirectory(src/lib)

# 5. Build the main executable (src/app)
add_subdirectory(src/app)

if(CONQUER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
