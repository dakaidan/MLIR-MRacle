FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG ENABLE_FUZZING=ON
ARG ENABLE_CUDA=OFF
ARG ENABLE_ROCM=OFF
ARG ENABLE_VULKAN=OFF

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    git \
    lld \
    ninja-build \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$ENABLE_FUZZING" = "ON" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        afl++ bison flex \
      && rm -rf /var/lib/apt/lists/*; \
    fi

RUN if [ "$ENABLE_CUDA" = "ON" ]; then \
      apt-get update && apt-get install -y --no-install-recommends nvidia-cuda-toolkit \
      && rm -rf /var/lib/apt/lists/*; \
    fi

RUN if [ "$ENABLE_ROCM" = "ON" ]; then \
      apt-get update && apt-get install -y --no-install-recommends rocm-dev \
      && rm -rf /var/lib/apt/lists/*; \
    fi

RUN if [ "$ENABLE_VULKAN" = "ON" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        libvulkan-dev vulkan-tools \
      && rm -rf /var/lib/apt/lists/*; \
    fi

ARG BUILD_CONQUER_OPT=ON

WORKDIR /workspace
COPY . .

RUN cmake -G Ninja \
      -S . \
      -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCONQUER_ENABLE_IREE=ON \
      -DCONQUER_BUILD_TESTS=OFF \
      -DCONQUER_BACKEND_CUDA=${ENABLE_CUDA} \
      -DCONQUER_BACKEND_ROCM=${ENABLE_ROCM} \
      -DCONQUER_BACKEND_VULKAN=${ENABLE_VULKAN}

RUN if [ "$BUILD_CONQUER_OPT" = "ON" ]; then \
      cmake --build build --target conquer-opt -j "$(nproc)"; \
    else \
      cmake --build build --target mlir-mr-smoke -j "$(nproc)"; \
    fi

ENV PATH="/workspace/build:/workspace/build/bin:${PATH}" \
    BUILD_CONQUER_OPT=${BUILD_CONQUER_OPT} \
    ENABLE_FUZZING=${ENABLE_FUZZING}

CMD ["./fuzzing/smoke-test.sh"]
