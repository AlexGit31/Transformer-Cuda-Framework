#include "RMSNormLayer.cuh"
#include <cmath>

__global__ void rmsnorm_forward_kernel(float* X, float* Y, float* gamma, float* inv_rms, int emb_dim, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float shared_sq[];
    float val = X[row * emb_dim + tid];
    shared_sq[tid] = val * val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) shared_sq[tid] += shared_sq[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        shared_sq[0] = rsqrtf((shared_sq[0] / emb_dim) + eps);
        inv_rms[row] = shared_sq[0];
    }
    __syncthreads();
    Y[row * emb_dim + tid] = val * shared_sq[0] * gamma[tid];
}

__global__ void rmsnorm_backward_kernel(float* dY, float* X, float* dX, float* inv_rms, float* gamma, float* dgamma, int emb_dim) {
    int row = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float shared_dot[];
    float dy = dY[row * emb_dim + tid];
    float x  = X[row * emb_dim + tid];
    float g  = gamma[tid];
    shared_dot[tid] = dy * x * g;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) shared_dot[tid] += shared_dot[tid + s];
        __syncthreads();
    }
    float dot_sum = shared_dot[0];
    float inv = inv_rms[row];
    float scale = (inv * inv * inv) / emb_dim;
    // Gradient for X
    dX[row * emb_dim + tid] = dy * inv * g - x * dot_sum * scale;
    // FIX #2: Gradient for gamma
    dgamma[tid] = dy * x * inv;
}

// Adam kernel for 1D parameter (gamma)
__global__ void gamma_adam_kernel(float* gamma, float* dgamma, float* m, float* v,
                                   float lr, int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        m[idx] = b1 * m[idx] + (1.0f - b1) * dgamma[idx];
        v[idx] = b2 * v[idx] + (1.0f - b2) * (dgamma[idx] * dgamma[idx]);
        float m_hat = m[idx] / (1.0f - powf(b1, (float)t));
        float v_hat = v[idx] / (1.0f - powf(b2, (float)t));
        gamma[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
        dgamma[idx] = 0.0f;
    }
}

RMSNormLayer::RMSNormLayer(int bs, int cs, int ed) : Layer(bs, cs, ed) {
    batch_size = bs; context_size = cs; embedding_dim = ed;
    cudaMalloc(&d_gamma, sizeof(float) * ed);
    cudaMalloc(&d_dgamma, sizeof(float) * ed);
    cudaMalloc(&d_m_gamma, sizeof(float) * ed);
    cudaMalloc(&d_v_gamma, sizeof(float) * ed);
    cudaMalloc(&d_inv_rms, sizeof(float) * bs * cs);
    cudaMalloc(&d_dX, sizeof(float) * bs * cs * ed);
    cudaMalloc(&d_Y, sizeof(float) * bs * cs * ed);
    float* h_gamma = (float*)malloc(sizeof(float) * ed);
    for (int i = 0; i < ed; i++) h_gamma[i] = 1.0f;
    cudaMemcpy(d_gamma, h_gamma, sizeof(float) * ed, cudaMemcpyHostToDevice);
    cudaMemset(d_dgamma, 0, sizeof(float) * ed);
    cudaMemset(d_m_gamma, 0, sizeof(float) * ed);
    cudaMemset(d_v_gamma, 0, sizeof(float) * ed);
    free(h_gamma);
}

RMSNormLayer::~RMSNormLayer() {
    cudaFree(d_gamma); cudaFree(d_dgamma); cudaFree(d_m_gamma); cudaFree(d_v_gamma);
    cudaFree(d_inv_rms); cudaFree(d_dX); cudaFree(d_Y);
}

float* RMSNormLayer::forward(cublasHandle_t h, void* inp, int act) {
    d_X_cache = (float*)inp;
    int total = batch_size * context_size;
    rmsnorm_forward_kernel<<<total, embedding_dim, embedding_dim * sizeof(float)>>>(
        d_X_cache, d_Y, d_gamma, d_inv_rms, embedding_dim, 1e-5f);
    cudaDeviceSynchronize();
    return d_Y;
}

float* RMSNormLayer::backward(cublasHandle_t h, float* d_dY) {
    int total = batch_size * context_size;
    // Zero dgamma before accumulation
    cudaMemset(d_dgamma, 0, sizeof(float) * embedding_dim);
    rmsnorm_backward_kernel<<<total, embedding_dim, embedding_dim * sizeof(float)>>>(
        d_dY, d_X_cache, d_dX, d_inv_rms, d_gamma, d_dgamma, embedding_dim);
    cudaDeviceSynchronize();
    return d_dX;
}

void RMSNormLayer::step(float lr, int t) {
    // FIX #2: Update gamma with Adam!
    int threads = 256;
    int blocks = (embedding_dim + threads - 1) / threads;
    gamma_adam_kernel<<<blocks, threads>>>(d_gamma, d_dgamma, d_m_gamma, d_v_gamma, lr, t, embedding_dim);
}
