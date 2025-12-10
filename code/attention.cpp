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


    auto start = std::chrono::high_resolution_clock::now();

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
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "QK^T Time: " << duration.count() << " ms" << std::endl;


    start = std::chrono::high_resolution_clock::now();

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
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;

    std::cout << "Softmax Time: " << duration.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    
    // S * V
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < D; k++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++) {
                acc += S[flat_idx(i, j, N)] * V[flat_idx(j, k, D)];
            }
            Output[flat_idx(i, k, D)] = acc;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;

    std::cout << "SV Time: " << duration.count() << " ms" << std::endl;
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
            diff_idx = i;
        }
    }
    
    if (max_diff > tol) {
        std::cerr << "FAILED" << std::endl;
        return false;
    } else {
        std::cout << "PASSED" << std::endl;
        return true;
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "invalid input args" << std::endl;
        return 1;
    }

    std::string test_case = argv[1];
    std::string base_path = "data/" + test_case + "/";

    int N, D;
    std::ifstream meta_file(base_path + "meta.txt");
    if (!meta_file.is_open()) {
        std::cerr << "invalid test case" << base_path << std::endl;
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

    std::cout << "Running Sequential Attention (N=" << N << ", D=" << D << ")..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    attention_sequential(Q.data(), K.data(), V.data(), Output.data(), N, D);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Total Time: " << duration.count() << " ms" << std::endl;

    check_accuracy(Output, RefOutput);

    return 0;
}
