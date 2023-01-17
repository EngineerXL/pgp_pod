# checkpgp
# NVCC = /usr/local/cuda/bin/nvcc
# MPIC++ = /usr/local/bin/mpic++

# Local
NVCC = nvcc

FLAGS_NVCC = -O2 -std=c++11 -Werror cross-execution-space-call -lm

BIN = cw_pod
# Compile solution
$(BIN): main.cu src/io.cuh src/polygon.cuh src/ray_cpu.cuh src/ray_gpu.cuh src/ssaa.cuh src/textures.cuh src/utils.cuh src/variables_cpu.cuh src/variables_gpu.cuh src/vector3d.cuh
	$(NVCC) $(FLAGS_NVCC) main.cu -o $(BIN)

MAX_FRAMES = 8192
# Run solution and make animation
run_gpu: $(BIN)
	bash scripts/run.sh $(BIN) $(MAX_FRAMES) "--gpu"

run_cpu: $(BIN)
	bash scripts/run.sh $(BIN) $(MAX_FRAMES) "--cpu"

clean:
	rm -rf *.o $(BIN)
