#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits> 
#include <fstream>
#include <string>
#include <iomanip>


// Flattens 2D index to 1D
inline int flat_idx(int row, int col, int dim) {
    return row * dim + col;
}

void attention_sequential(
    const float* Q, const float* K, const float* V, 
    float* Output, 
    int N, int D
) {
    std::vector<float> S(N * N);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // QK^T
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < D; k++) {
                dot += Q[flat_idx(i, k, D)] * K[flat_idx(j, k, D)];
            }
            S[flat_idx(i, j, N)] = dot * scale;
        }
    }

    // Softmax
    for (int i = 0; i < N; i++) {
        // Find Max 
        float max_val = std::numeric_limits<float>::min();
        for (int j = 0; j < N; j++) {
            if (S[flat_idx(i, j, N)] > max_val) {
                max_val = S[flat_idx(i, j, N)];
            }
        }

        // Exponentiate and Sum 
        float sum_exp = 0.0f;
        for (int j = 0; j < N; j++) {
            float val = std::exp(S[flat_idx(i, j, N)] - max_val);
            S[flat_idx(i, j, N)] = val; // Store exp value temporarily
            sum_exp += val;
        }

        // Normalize
        for (int j = 0; j < N; j++) {
            S[flat_idx(i, j, N)] /= sum_exp;
        }
    }

    // S * V ---
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < D; k++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++) {
                acc += S[flat_idx(i, j, N)] * V[flat_idx(j, k, D)];
            }
            Output[flat_idx(i, k, D)] = acc;
        }
    }
}

// Inititalize random data
void init_random(std::vector<float>& vec) {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-1, 1);
    for (auto& v : vec) v = dis(e);
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
    int diff_idx = -1;

    for (size_t i = 0; i < ours.size(); i++) {
        float diff = std::abs(ours[i] - ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
            diff_idx = i;
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


int main() {
    int N = 1024; 
    int D = 64;

    // Allocate memory
    std::vector<float> Q(N * D);
    std::vector<float> K(N * D);
    std::vector<float> V(N * D);
    std::vector<float> Output(N * D);
    std::vector<float> RefOutput(N * D);

    load_binary("data/q.bin", Q);
    load_binary("data/k.bin", K);
    load_binary("data/v.bin", V);
    load_binary("data/out_ref.bin", RefOutput);

    std::cout << "Running Sequential Attention (N=" << N << ", D=" << D << ")..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    attention_sequential(Q.data(), K.data(), V.data(), Output.data(), N, D);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Time: " << duration.count() << " ms" << std::endl;

    check_accuracy(Output, RefOutput);

    return 0;
}