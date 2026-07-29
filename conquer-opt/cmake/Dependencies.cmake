include(FetchContent)

# --- JSON ---
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "Fetching nlohmann_json...")
    FetchContent_Declare(nlohmann_json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.10.5)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# --- CLI11 ---
find_package(CLI11 QUIET)
if(NOT CLI11_FOUND)
    message(STATUS "Fetching CLI11...")
    FetchContent_Declare(CLI11 GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git GIT_TAG v2.4.2)
    FetchContent_MakeAvailable(CLI11)
endif()

option(CONQUER_BUILD_TESTS "Build the ConQuER test suite" ON)

if(CONQUER_BUILD_TESTS)
    message(STATUS "Fetching GoogleTest...")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

# --- Eigen ---
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Fetching Eigen...")
    FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG        3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()