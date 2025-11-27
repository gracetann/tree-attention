#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>   
#include <string>  
#include <chrono>

__device__ inline int idx(int row, int col, int dim) {
    return row * dim + col;
}

// Compute Q * K^T 
__global__ void kernel_QKT(const float* Q, const float* K, float* S, int N, int D, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float val = 0.0f;
        for (int d = 0; d < D; ++d) {
            val += Q[idx(row, d, D)] * K[idx(col, d, D)];
        }
        S[idx(row, col, N)] = val * scale;
    }
}

__global__ void kernel_softmax(float* S, int N) {
    extern __shared__ float sdata[]; 

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_max = -1e20f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = S[idx(row, i, N)];
        if (val > local_max) {
            local_max = val;
        }
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }
        __syncthreads();
    }
    float row_max = sdata[0]; 
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = S[idx(row, i, N)];
        local_sum += expf(val - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        float val = S[idx(row, i, N)];
        S[idx(row, i, N)] = expf(val - row_max) / row_sum;
    }
}

// compute S * V 
__global__ void kernel_apply_attention(const float* S, const float* V, float* O, int N, int D) {
    int row = blockIdx.y;
    int col = blockIdx.x;

    if (row < N && col < D) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            acc += S[idx(row, k, N)] * V[idx(k, col, D)];
        }
        O[idx(row, col, D)] = acc;
    }
}

void attention(const float* h_Q, const float* h_K, const float* h_V, float* h_O, int N, int D) {
    float *d_Q, *d_K, *d_V, *d_S, *d_O;
    size_t size_mat = N * D * sizeof(float);
    size_t size_scores = N * N * sizeof(float);

    cudaMalloc(&d_Q, size_mat);
    cudaMalloc(&d_K, size_mat);
    cudaMalloc(&d_V, size_mat);
    cudaMalloc(&d_O, size_mat);
    cudaMalloc(&d_S, size_scores);

    cudaMemcpy(d_Q, h_Q, size_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_mat, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Scores
    float scale = 1.0f / sqrtf((float)D);
    kernel_QKT<<<numBlocks, threadsPerBlock>>>(d_Q, d_K, d_S, N, D, scale);
    cudaDeviceSynchronize();

    // Softmax (Tree Reduction)
    int threads_reduction = 1024; 
    size_t shared_mem_size = threads_reduction * sizeof(float);
    kernel_softmax<<<N, threads_reduction, shared_mem_size>>>(d_S, N);
    cudaDeviceSynchronize();

    dim3 blockOut(1, 1); // one thread per output element 
    dim3 gridOut(D, N);
    kernel_apply_attention<<<gridOut, blockOut>>>(d_S, d_V, d_O, N, D);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_O, d_O, size_mat, cudaMemcpyDeviceToHost);

    cudaFree(d_Q); 
    cudaFree(d_K); 
    cudaFree(d_V); 
    cudaFree(d_S); 
    cudaFree(d_O);
}

void load_binary(const std::string& filename, std::vector<float>& buffer) {
    std::ifstream file(filename, std::ios::binary);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
    file.close();
}

bool check_accuracy(const std::vector<float>& ours, const std::vector<float>& ref, float tol=1e-4) {
    if (ours.size() != ref.size()) {
        std::cerr << "Size mismatch - Ours: " << ours.size() << " Ref: " << ref.size() << std::endl;
        return false;
    }

    float max_diff = 0.0f;
    // int diff_idx = -1;

    for (size_t i = 0; i < ours.size(); i++) {
        float diff = std::abs(ours[i] - ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
            // diff_idx = i;
        }
    }

    // std::cout << "Max Absolute Error: " << max_diff << " at index " << diff_idx << std::endl;
    // std::cout << "  Ours: " << ours[diff_idx] << std::endl;
    // std::cout << "  Ref:  " << ref[diff_idx] << std::endl;

    if (max_diff > tol) {
        std::cerr << "FAILED (Tolerance: " << tol << ")" << std::endl;
        return false;
    } else {
        std::cout << "PASSED" << std::endl;
        return true;
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test_case_name>" << std::endl;
        return 1;
    }

    std::string test_case = argv[1];
    std::string base_path = "data/" + test_case + "/";

    // Read Dimensions
    int N, D;
    std::ifstream meta_file(base_path + "meta.txt");
    if (!meta_file.is_open()) {
        std::cerr << "Error: Could not open test case " << base_path << std::endl;
        return 1;
    }
    meta_file >> N >> D;
    meta_file.close();

    std::cout << "Test Case: " << test_case << " | N=" << N << " D=" << D << std::endl;

    std::vector<float> Q(N * D);
    std::vector<float> K(N * D);
    std::vector<float> V(N * D);
    std::vector<float> Output(N * D);
    std::vector<float> RefOutput(N * D);

    load_binary(base_path + "q.bin", Q);
    load_binary(base_path + "k.bin", K);
    load_binary(base_path + "v.bin", V);
    load_binary(base_path + "out_ref.bin", RefOutput);

    std::cout << "Running CUDA Attention (N=" << N << ", D=" << D << ")..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    attention(Q.data(), K.data(), V.data(), Output.data(), N, D);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Time: " << duration.count() << " ms" << std::endl;

    check_accuracy(Output, RefOutput);

    return 0;
}