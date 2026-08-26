# Build: docker build -t mlir-mracle .
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
    curl \
    git \
    lld \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

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

WORKDIR /workspace

COPY requirements.txt .
RUN uv python install 3.12 \
 && uv venv --python 3.12 \
 && uv pip install -r requirements.txt

COPY . .

RUN .venv/bin/cmake -G Ninja \
      -S . \
      -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_PREFIX_PATH=$(.venv/bin/python -m mlir_wheel --root-dir)

RUN .venv/bin/cmake --build build --target mlir_mracle_opt -j "$(nproc)"

ENV PATH="/workspace/.venv/bin:/workspace/build:/workspace/build/bin:${PATH}" \
    ENABLE_FUZZING=${ENABLE_FUZZING}

CMD ["./fuzzing/smoke-test.sh"]
