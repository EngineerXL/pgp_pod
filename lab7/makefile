FLAGS_G++ = -g -O2 -pedantic -std=c++17 -Wall -Wextra
FLAGS_MPICPP = -fopenmp -lm -O2 -pedantic --std=c++11 -Wall -Wextra -Werror -Wno-long-long -Wno-sign-compare

TESTS = 1
NP = 6
N = 1

# Compile solution
solution: main.cpp
	mpic++ $(FLAGS_MPICPP) main.cpp -o solution

# Run solution on sample
runs: solution
	mpirun -np $(N) ./solution < samples/$(N).in

# Run solution on test
runt: solution
	mpirun -np $(NP) ./solution < tests/$(N).in 


# Compile solution for answer generation
correct: correct.cpp
	g++ $(FLAGS_G++) correct.cpp -o correct

# Generate tests and answers based on correct solution
test: correct test_gen.py
	bash scripts/generate_tests.sh $(TESTS)
	python3 test_gen.py $(TESTS)
	# bash scripts/gen_ans.sh $(TESTS)

testrun: test solution
	mpirun -np $(NP) ./solution < tests/1.in 
	mpirun -np $(NP) ./solution < tests/2.in 
	mpirun -np $(NP) ./solution < tests/3.in 
	mpirun -np $(NP) ./solution < tests/4.in 
	mpirun -np $(NP) ./solution < tests/5.in

# Check solution
check: solution
	bash scripts/checker.sh $(TESTS) $(NP)

clean:
	rm -rf *.o solution correct bench
