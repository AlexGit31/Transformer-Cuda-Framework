#include "RMSNormLayer.cuh"
#include <iostream>
#include <cmath>

// =========================================================================
// KERNELS CUDA
// =========================================================================

__global__ void rmsnorm_forward_kernel(float* d_X, float* d_Y, float* d_gamma, float* d_inv_rms, int embedding_dim, float epsilon) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float shared_sq_sum[];
    
    float val = d_X[row * embedding_dim + tid];
    shared_sq_sum[tid] = val * val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        shared_sq_sum[0] = rsqrtf((shared_sq_sum[0] / embedding_dim) + epsilon); 
        d_inv_rms[row] = shared_sq_sum[0]; 
    }
    __syncthreads();
    
    float inv_rms = shared_sq_sum[0];
    d_Y[row * embedding_dim + tid] = val * inv_rms * d_gamma[tid];
}

// =========================================================================
// MÉTHODES DE LA CLASSE
// =========================================================================

// L'implémentation exacte qui correspond au .cuh
RMSNormLayer::RMSNormLayer(int batch_size, int context_size, int embedding_dim) 
    : Layer(batch_size, context_size, embedding_dim) {
    
    this->batch_size = batch_size;
    this->context_size = context_size;
    this->embedding_dim = embedding_dim;

    cudaMalloc(&d_gamma, sizeof(float) * embedding_dim);
    cudaMalloc(&d_inv_rms, sizeof(float) * batch_size * context_size);
    cudaMalloc(&d_dX, sizeof(float) * batch_size * context_size * embedding_dim);
    cudaMalloc(&d_Y, sizeof(float) * batch_size * context_size * embedding_dim);
    float* h_gamma = (float*)malloc(sizeof(float) * embedding_dim);
    for(int i = 0; i < embedding_dim; i++) h_gamma[i] = 1.0f;
    cudaMemcpy(d_gamma, h_gamma, sizeof(float) * embedding_dim, cudaMemcpyHostToDevice);
    free(h_gamma);
}

RMSNormLayer::~RMSNormLayer() {
    cudaFree(d_gamma);
    cudaFree(d_inv_rms);
    cudaFree(d_dX);
    cudaFree(d_Y);
}

float* RMSNormLayer::forward(cublasHandle_t handle, void* d_input, int activation_type) {
    float* d_X = (float*) d_input;
    this->d_X_cache = d_X;
    
    int total_words = batch_size * context_size;
    int threadsPerBlock = embedding_dim;
    size_t shared_mem = embedding_dim * sizeof(float);
    
    rmsnorm_forward_kernel<<<total_words, threadsPerBlock, shared_mem>>>(
        d_X, d_Y, d_gamma, d_inv_rms, embedding_dim, 1e-5f
    );
    
    cudaDeviceSynchronize();
    return d_Y;
}

// Le VRAI Kernel Backward de la RMSNorm (Mathématiquement exact)
__global__ void rmsnorm_backward_kernel(float* d_dY, float* d_X, float* d_dX, float* d_inv_rms, int embedding_dim) {
    int row = blockIdx.x; // 1 bloc = 1 mot
    int tid = threadIdx.x;

    extern __shared__ float shared_dot[];

    float dy = d_dY[row * embedding_dim + tid];
    float x  = d_X[row * embedding_dim + tid];

    // 1. On calcule le produit scalaire (dY * X)
    shared_dot[tid] = dy * x;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
        }
        __syncthreads();
    }

    // 2. On applique la vraie formule de la dérivée
    float dot_sum = shared_dot[0];
    float inv = d_inv_rms[row];
    float scale = (inv * inv * inv) / embedding_dim;

    // La force de rappel magique est ce signe "moins" !
    d_dX[row * embedding_dim + tid] = (dy * inv) - (x * dot_sum * scale);
}

float* RMSNormLayer::backward(cublasHandle_t handle, float* d_dY) {
    int total_words = batch_size * context_size;
    int threadsPerBlock = embedding_dim;
    size_t shared_mem = embedding_dim * sizeof(float);
    
    // On appelle notre nouveau kernel ultra-robuste
    rmsnorm_backward_kernel<<<total_words, threadsPerBlock, shared_mem>>>(
        d_dY, d_X_cache, d_dX, d_inv_rms, embedding_dim
    );
    
    cudaDeviceSynchronize();
    return d_dX; 
}

void RMSNormLayer::step(float learning_rate, int t) {
    // Vide, Gamma reste figé
}
