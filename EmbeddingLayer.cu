#include "EmbeddingLayer.cuh"
#include <cmath>
#include <cstdlib>

// Re-use the xavier init kernel (declare extern)
__global__ void xavier_init_kernel(float* W, int in_f, int out_f, unsigned int seed);

__global__ void embedding_forward_kernel(int* X, float* W, float* PE, float* Y, int emb_dim, int cs, int total) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total) {
        int e_idx = idx % emb_dim;
        int w_pos = idx / emb_dim;
        int s_pos = w_pos % cs;
        int v_id = X[w_pos];
        Y[idx] = W[v_id * emb_dim + e_idx] + PE[s_pos * emb_dim + e_idx];
    }
}

__global__ void backward_embedding_kernel(int* X, float* dY, float* dW, int emb_dim, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int e_idx = idx % emb_dim;
        int w_pos = idx / emb_dim;
        int v_id = X[w_pos];
        atomicAdd(&dW[v_id * emb_dim + e_idx], dY[idx]);
    }
}

__global__ void embedding_adam_kernel(float* W, float* dW, float* m, float* v, float lr, int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
        m[idx] = b1 * m[idx] + (1.0f - b1) * dW[idx];
        v[idx] = b2 * v[idx] + (1.0f - b2) * (dW[idx] * dW[idx]);
        float m_hat = m[idx] / (1.0f - powf(b1, (float)t));
        float v_hat = v[idx] / (1.0f - powf(b2, (float)t));
        W[idx] -= lr * wd * W[idx];
        W[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
        dW[idx] = 0.0f;
    }
}

EmbeddingLayer::EmbeddingLayer(int vs, int ed, int bs, int cs) : Layer(bs, cs, ed) {
    vocab_size = vs; embedding_dim = ed; batch_size = bs; context_size = cs;
    cudaMalloc(&d_W, sizeof(float) * vs * ed);
    cudaMalloc(&d_dW, sizeof(float) * vs * ed);
    cudaMalloc(&d_Y, sizeof(float) * bs * cs * ed);
    cudaMalloc(&d_PE, sizeof(float) * cs * ed);

    // FIX #3: Xavier init on GPU
    int threads = 256;
    int blocks = (vs * ed + threads - 1) / threads;
    xavier_init_kernel<<<blocks, threads>>>(d_W, vs, ed, 42);

    // Positional encoding on CPU
    float* h_PE = (float*)malloc(sizeof(float) * cs * ed);
    for (int pos = 0; pos < cs; pos++) {
        for (int i = 0; i < ed; i += 2) {
            float div = pow(10000.0f, (float)i / ed);
            h_PE[pos * ed + i] = sin(pos / div);
            if (i + 1 < ed) h_PE[pos * ed + i + 1] = cos(pos / div);
        }
    }
    cudaMemcpy(d_PE, h_PE, sizeof(float) * cs * ed, cudaMemcpyHostToDevice);
    free(h_PE);

    cudaMalloc(&d_m, sizeof(float) * vs * ed);
    cudaMalloc(&d_v, sizeof(float) * vs * ed);
    cudaMemset(d_m, 0, sizeof(float) * vs * ed);
    cudaMemset(d_v, 0, sizeof(float) * vs * ed);
}

EmbeddingLayer::~EmbeddingLayer() {
    cudaFree(d_W); cudaFree(d_dW); cudaFree(d_Y); cudaFree(d_PE); cudaFree(d_m); cudaFree(d_v);
}

float* EmbeddingLayer::forward(cublasHandle_t h, void* inp, int act) {
    d_X = (int*)inp;
    int total = batch_size * context_size;
    int tpb = 256;
    int bpg = (total + tpb - 1) / tpb;
    embedding_forward_kernel<<<bpg, tpb>>>(d_X, d_W, d_PE, d_Y, embedding_dim, context_size, total);
    cudaDeviceSynchronize();
    return d_Y;
}

float* EmbeddingLayer::backward(cublasHandle_t h, float* d_dY) {
    int total = batch_size * context_size * embedding_dim;
    int tpb = 256;
    int bpg = (total + tpb - 1) / tpb;
    cudaMemset(d_dW, 0, sizeof(float) * vocab_size * embedding_dim);
    backward_embedding_kernel<<<bpg, tpb>>>(d_X, d_dY, d_dW, embedding_dim, total);
    cudaDeviceSynchronize();
    return nullptr;
}

void EmbeddingLayer::step(float lr, int t) {
    int total = vocab_size * embedding_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    embedding_adam_kernel<<<blocks, threads>>>(d_W, d_dW, d_m, d_v, lr, t, total);
}
