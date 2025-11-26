### 1. Generate Input Data
Run the Python script **locally** to generate random matrices (Q, K, V) and the expected output using PyTorch. This will create a `data/` directory containing binary files.

```bash
python3 attn_verify.py
```

### 2\. Compile `attention.cpp`

```bash
g++ -O3 -o attention attention.cpp
```

### 3\. Run and Verify

Run the executable. It will load the binary files from `data/`, run the sequential attention algorithm, and compare the result against the PyTorch reference.

```bash
./attention
```

**Expected Output:**

```text
Running Sequential Attention (N=1024, D=64)...
Time: [X] ms
PASSED
```
