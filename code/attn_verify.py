import torch
import numpy as np
import torch.nn.functional as F
import os

def generate_test_data(N=1024, D=64):
    print(f"Generating data for N={N}, D={D}")
    
    torch.manual_seed(42) # Fixed seed for reproducibility
    Q = torch.randn(1, 1, N, D)
    K = torch.randn(1, 1, N, D)
    V = torch.randn(1, 1, N, D)

    expected_output = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

    if not os.path.exists("data"):
        os.makedirs("data")

    Q.numpy().flatten().astype(np.float32).tofile("data/q.bin")
    K.numpy().flatten().astype(np.float32).tofile("data/k.bin")
    V.numpy().flatten().astype(np.float32).tofile("data/v.bin")
    expected_output.numpy().flatten().astype(np.float32).tofile("data/out_ref.bin")
    
    print(f"Saved binary files to ./data/")
    print(f"Sample Ref Value [0]: {expected_output.flatten()[0]:.6f}")

if __name__ == "__main__":
    generate_test_data()