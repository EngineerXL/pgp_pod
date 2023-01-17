FLAGS_G++ = -g -O2 -pedantic -std=c++17 -Wall -Wextra
FLAGS_NVCC = -g -O2 -std=c++11

TESTS = 10

# Compile solution
solution: main.cu algo.cuh ../cuda_utils.cuh
	nvcc $(FLAGS_NVCC) main.cu -o solution

# Compile benchmark file
bench: bench.cu algo.cuh ../cuda_utils.cuh
	nvcc $(FLAGS_NVCC) bench.cu -o bench

# Run solution on all tests
runall: solution
	bash scripts/run_tests.sh "solution" $(TESTS)

# Run benchmark on all tests
brunall: bench
	bash scripts/run_tests.sh "bench" $(TESTS)

# Compile solution for answer generation
correct: correct.cpp
	g++ $(FLAGS_G++) correct.cpp -o correct

# Generate tests and answers based on correct solution
test: correct test_gen.py
	bash scripts/generate_tests.sh $(TESTS)
	python3 test_gen.py
	bash scripts/gen_ans.sh $(TESTS)

# Check solution
check: solution
	bash scripts/checker.sh $(TESTS)

clean:
	rm -rf *.o solution correct bench
