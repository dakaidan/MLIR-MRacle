# Only compile the main.cpp file
add_executable(conquer-opt main.cpp)

target_compile_options(conquer-opt PRIVATE -fno-rtti)
set_target_properties(conquer-opt PROPERTIES
        ENABLE_EXPORTS TRUE
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Link our core library. Since it was linked PUBLICLY, conquer-opt
# automatically inherits all the MLIR includes, macros, and dependencies.
target_link_libraries(conquer-opt PRIVATE conquer_lib)

# Installation
install(TARGETS conquer-opt
        DESTINATION bin
        COMPONENT conquer
)