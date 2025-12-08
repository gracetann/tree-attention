### Generate Input Data
Run the Python script **locally** to generate the test suite of random matrices (Q, K, V) and the expected output using PyTorch. This will create a `data/` directory containing the test cases. 

```bash
mkdir data
python3 gen_tests.py
```

### Compile Serial Implementation

```bash
g++ -O3 -o attention_serial attention.cpp
```

### Run and Verify

Run the executable. It will load the binary files from `data/`, run the sequential attention algorithm, and compare the result against the PyTorch reference.

```bash
./attention_serial [TEST_CASE_NAME]
```

**Expected Output:**

```text
Running Sequential Attention (N=1024, D=64)...
Time: [X] ms
PASSED
```

### Compile CUDA Program
```bash
export PATH=/usr/local/cuda-11.7/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64/:${LD_LIBRARY_PATH}
nvcc -o attention attention.cu -O3 -ccbin /usr/bin/g++-11
```

### Run and Verify
Run the CUDA executable on the generated tests:
```bash
./attention [TEST_CASE_NAME]
```

### NCU Kernel Profiling

```bash
nvcc -o attention attention.cu -O3 -lineinfo -ccbin /usr/bin/g++-11
ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all ./attention n4096_d64
```
