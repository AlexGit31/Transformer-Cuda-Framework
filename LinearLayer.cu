#include "LinearLayer.cuh"
#include <iostream>
__global__ void linear_adam_kernel(float* W, float* dW, float* m, float* v, float lr, int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;

        m[idx] = beta1 * m[idx] + (1.0f - beta1) * dW[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * (dW[idx] * dW[idx]);

        float m_hat = m[idx] / (1.0f - powf(beta1, (float)t));
        float v_hat = v[idx] / (1.0f - powf(beta2, (float)t));

        float weight_decay = 0.01f;
        W[idx] -= lr * weight_decay * W[idx];

        W[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        dW[idx] = 0.0f; // Remise à zéro du gradient !
    }
}
// Kernel pour ajouter les biais
__global__ void add_bias_kernel(float* d_Y, const float* d_b, int batch_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_features * out_features;
    if (idx < total_elements) {
        int col = idx % out_features;
        d_Y[idx] += d_b[col];
    }
}

// Kernel pour la descente de gradient
__global__ void sgd_update_linear(float* param, const float* grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param[idx] -= lr * grad[idx];
    }
}

LinearLayer::LinearLayer(int batch_size, int in_feat, int out_feat) 
    : Layer(batch_size, in_feat, out_feat) {
    
    // Allocations VRAM
    cudaMalloc(&d_W, sizeof(float) * in_feat * out_feat);
    cudaMalloc(&d_b, sizeof(float) * out_feat);
    cudaMalloc(&d_dW, sizeof(float) * in_feat * out_feat);
    cudaMalloc(&d_db, sizeof(float) * out_feat);
    cudaMalloc(&d_Y, sizeof(float) * batch_size * out_feat);
    cudaMalloc(&d_dX, sizeof(float) * batch_size * in_feat);

    // Initialisation aléatoire des poids (CPU -> GPU)
    // Initialisation symétrique (entre -0.05 et +0.05) centrée sur 0 !
    float* h_W = (float*)malloc(sizeof(float) * in_feat * out_feat);
    for(int i = 0; i < in_feat * out_feat; i++) {
        // (rand() / RAND_MAX) donne entre 0 et 1. 
        // * 2.0 - 1.0 donne entre -1 et 1.
        h_W[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.05f; 
    }
    cudaMemcpy(d_W, h_W, sizeof(float) * in_feat * out_feat, cudaMemcpyHostToDevice);
    
    cudaMemset(d_b, 0, sizeof(float) * out_feat); // Biais à zéro
    
    cudaMalloc(&d_m, sizeof(float) * in_feat * out_feat);
    cudaMalloc(&d_v, sizeof(float) * in_feat * out_feat);
    cudaMemset(d_m, 0, sizeof(float) * in_feat * out_feat);
    cudaMemset(d_v, 0, sizeof(float) * in_feat * out_feat);

    free(h_W);
}

LinearLayer::~LinearLayer() {
    cudaFree(d_W); cudaFree(d_b); cudaFree(d_dW);
    cudaFree(d_db); cudaFree(d_Y); cudaFree(d_dX);
    cudaFree(d_m);
    cudaFree(d_v);
}

float* LinearLayer::forward(cublasHandle_t handle, void* d_input, int activation_type) {
    d_X_cache = (float*)d_input; // On sauvegarde l'entrée pour le backward !
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // d_Y = d_X * d_W
    // Rappel cuBLAS (Column-Major) : on calcule W^T * X^T pour obtenir Y^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                out_features, batch_features, in_features,
                &alpha,
                d_W, out_features,
                d_X_cache, in_features,
                &beta,
                d_Y, out_features);

    // Ajout des biais
    int threads = 256;
    int blocks = (batch_features * out_features + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(d_Y, d_b, batch_features, out_features);
    
    // (Dans un code complet, on appliquerait le ReLU ici si activation_type == ACTIVATION_RELU)

    return d_Y;
}

float* LinearLayer::backward(cublasHandle_t handle, float* d_dY) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 1. Calcul de dX = dY * W^T (Pour la couche d'en dessous)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                in_features, batch_features, out_features,
                &alpha, d_W, out_features, d_dY, out_features,
                &beta, d_dX, in_features);

    // 2. Calcul de dW = X^T * dY (Pour mettre à jour nos propres poids)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                out_features, in_features, batch_features,
                &alpha, d_dY, out_features, d_X_cache, in_features,
                &beta, d_dW, out_features);
    return d_dX;
}

void LinearLayer::step(float learning_rate, int t) {
    int total_elements = in_features * out_features;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    linear_adam_kernel<<<blocks, threads>>>(d_W, d_dW, d_m, d_v, learning_rate, t, total_elements);
}
