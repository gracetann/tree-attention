import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

def generate_test_case(N, D):
    # naming convention: n{N}_d{D}
    base_dir="data"
    test_case_name = f"n{N}_d{D}"
    case_dir = os.path.join(base_dir, test_case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    print(f"Generating {test_case_name} (N={N}, D={D})...")
    
    torch.manual_seed(42)
    
    # Generate Tensors
    Q = torch.randn(1, 1, N, D)
    K = torch.randn(1, 1, N, D)
    V = torch.randn(1, 1, N, D)
    
    # Compute Reference Output
    ref_out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    
    Q.numpy().flatten().astype(np.float32).tofile(os.path.join(case_dir, "q.bin"))
    K.numpy().flatten().astype(np.float32).tofile(os.path.join(case_dir, "k.bin"))
    V.numpy().flatten().astype(np.float32).tofile(os.path.join(case_dir, "v.bin"))
    ref_out.numpy().flatten().astype(np.float32).tofile(os.path.join(case_dir, "out_ref.bin"))
    
    # Save Metadata 
    with open(os.path.join(case_dir, "meta.txt"), "w") as f:
        f.write(f"{N} {D}")

def main():
    test_suite = [
        (256, 64),
        (512, 64),
        (1024, 64),
        (2048, 64),
        (4096, 64),
        (8192, 64)  
    ]
    
    for N, D in test_suite:
        generate_test_case(N, D)
        
if __name__ == "__main__":
    main()