#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>   
#include <string>  
#include <chrono>

#define TILE_SIZE 16

__device__ inline int idx(int row, int col, int dim) {
    return row * dim + col;
}

__global__ void kernel_QKT_tiled(const float* Q, const float* K, float* S, int N, int D, float scale) {
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float val = 0.0f;

    for (int i = 0; i < (D + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (row < N && i * TILE_SIZE + tx < D) {
            tile_Q[ty][tx] = Q[idx(row, i * TILE_SIZE + tx, D)];
        } else {
            tile_Q[ty][tx] = 0.0f;
        }

        if (col < N && i * TILE_SIZE + ty < D) {
             tile_K[tx][ty] = K[idx(col, i * TILE_SIZE + ty, D)];
        } else {
            tile_K[tx][ty] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            if (i * TILE_SIZE + j < D) {
                val += tile_Q[ty][j] * tile_K[tx][j];
            }
        }
        
        __syncthreads();
    }

    if (row < N && col < N) {
        S[idx(row, col, N)] = val * scale;
    }
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

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vsmem = sdata; 
        
        if (blockDim.x >= 64 && vsmem[tid + 32] > vsmem[tid]) {
            vsmem[tid] = vsmem[tid + 32];
        }
        if (vsmem[tid + 16] > vsmem[tid]) vsmem[tid] = vsmem[tid + 16];
        if (vsmem[tid + 8] > vsmem[tid])  vsmem[tid] = vsmem[tid + 8];
        if (vsmem[tid + 4] > vsmem[tid])  vsmem[tid] = vsmem[tid + 4];
        if (vsmem[tid + 2] > vsmem[tid])  vsmem[tid] = vsmem[tid + 2];
        if (vsmem[tid + 1] > vsmem[tid])  vsmem[tid] = vsmem[tid + 1];
    }
    
    __syncthreads();
    float row_max = sdata[0]; 
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = S[idx(row, i, N)];
        local_sum += expf(val - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vsmem = sdata;
        
        if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    __syncthreads();
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        float val = S[idx(row, i, N)];
        S[idx(row, i, N)] = expf(val - row_max) / row_sum;
    }
}

__global__ void kernel_softmax_naive(float* S, int N) {
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

    if (tid == 0) {
        float row_max = -1e20f;
        for (int i = 0; i < blockDim.x; i++) {
            if (sdata[i] > row_max) {
                row_max = sdata[i];
            }
        }
        sdata[0] = row_max;
    }
    __syncthreads();

    float row_max = sdata[0]; 

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = S[idx(row, i, N)];
        local_sum += expf(val - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float row_sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            row_sum += sdata[i];
        }
        sdata[0] = row_sum;
    }
    __syncthreads();
    
    float row_sum = sdata[0];

    for (int i = tid; i < N; i += blockDim.x) {
        float val = S[idx(row, i, N)];
        S[idx(row, i, N)] = expf(val - row_max) / row_sum;
    }
}

// compute S * V 
__global__ void kernel_attention(const float* S, const float* V, float* O, int N, int D) {
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

__global__ void kernel_attention_tiled(const float* S, const float* V, float* O, int N, int D) {
    __shared__ float tile_S[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_V[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty; 
    int col = bx * TILE_SIZE + tx;

    float val = 0.0f;
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        
        // load tile into shared memory
        if (row < N && (m * TILE_SIZE + tx) < N) {
            tile_S[ty][tx] = S[idx(row, m * TILE_SIZE + tx, N)];
        } else {
            tile_S[ty][tx] = 0.0f;
        }

        if ((m * TILE_SIZE + ty) < N && col < D) {
            tile_V[ty][tx] = V[idx(m * TILE_SIZE + ty, col, D)];
        } else {
            tile_V[ty][tx] = 0.0f;
        }

        __syncthreads(); 

        // compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            val += tile_S[ty][k] * tile_V[k][tx];
        }
        
        __syncthreads(); 
    }

    if (row < N && col < D) {
        O[idx(row, col, D)] = val;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    // Scores
    float scale = 1.0f / sqrtf((float)D);
    kernel_QKT_tiled<<<numBlocks, threadsPerBlock>>>(d_Q, d_K, d_S, N, D, scale);
    cudaDeviceSynchronize();

    cudaEvent_t softmax_start, softmax_stop;
    cudaEventCreate(&softmax_start);
    cudaEventCreate(&softmax_stop);

    // Softmax (Tree Reduction)
    int threads_reduction = 1024; 
    size_t shared_mem_size = threads_reduction * sizeof(float);

    cudaEventRecord(softmax_start);
    kernel_softmax<<<N, threads_reduction, shared_mem_size>>>(d_S, N);
    cudaEventRecord(softmax_stop);
    cudaEventSynchronize(softmax_stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, softmax_start, softmax_stop);

    std::cout << "Softmax Execution Time: " << milliseconds << " ms" << std::endl;

    // dim3 blockOut(1, 1); // one thread per output element 
    // dim3 gridOut(D, N);
    // kernel_attention<<<gridOut, blockOut>>>(d_S, d_V, d_O, N, D);
    dim3 numBlocksAttn((D + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    kernel_attention_tiled<<<numBlocksAttn, threadsPerBlock>>>(d_S, d_V, d_O, N, D);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0;
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

    std::cout << "Total Execution Time: " << duration.count() << " ms" << std::endl;

    check_accuracy(Output, RefOutput);

    return 0;
}