#include "LinearLayer.cuh"
#include <cmath>
#include <cstdlib>

// ─── Xavier/Glorot uniform initialization ──────────────────────────────
__global__ void xavier_init_kernel(float* W, int in_f, int out_f, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in_f * out_f;
    if (idx < total) {
        float limit = sqrtf(6.0f / (float)(in_f + out_f));
        // Simple pseudo-random on GPU using thread index + seed
        unsigned int s = idx + seed;
        s = (s ^ 61) ^ (s >> 16);
        s = s * 9;
        s = s ^ (s >> 4);
        s = s * 0x27d4eb2d;
        s = s ^ (s >> 15);
        float r = (float)(s & 0xFFFF) / 65535.0f;
        W[idx] = (2.0f * r - 1.0f) * limit;
    }
}

// ─── FIX #1: ReLU activation kernel ────────────────────────────────────
__global__ void relu_forward_kernel(float* d_Y, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        if (d_Y[idx] < 0.0f) d_Y[idx] = 0.0f;
    }
}

__global__ void relu_backward_kernel(float* d_dY, float* d_Y, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        if (d_Y[idx] <= 0.0f) d_dY[idx] = 0.0f;
    }
}

// ─── FIX #5: GELU activation kernel (approximation) ────────────────────
__global__ void gelu_forward_kernel(float* d_Y, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float x = d_Y[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        d_Y[idx] = x * cdf;
    }
}

__global__ void gelu_backward_kernel(float* d_dY, float* d_Y_save, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float x = d_Y_save[idx];
        float x3 = x * x * x;
        float tanh_arg = 0.7978845608f * (x + 0.044715f * x3);
        float tanh_val = tanhf(tanh_arg);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float gelu_deriv = 0.5f * (1.0f + tanh_val) + 0.5f * x * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x) * sech2;
        d_dY[idx] *= gelu_deriv;
    }
}

// ─── AdamW kernel ───────────────────────────────────────────────────────
__global__ void linear_adam_kernel(float* W, float* dW, float* m, float* v, float lr, int t, int size) {
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

__global__ void add_bias_kernel(float* Y, const float* b, int batch_f, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_f * out_f) {
        Y[idx] += b[idx % out_f];
    }
}

// ─── Constructor with Xavier init ───────────────────────────────────────
LinearLayer::LinearLayer(int bs, int in_f, int out_f) : Layer(bs, in_f, out_f) {
    cudaMalloc(&d_W, sizeof(float) * in_f * out_f);
    cudaMalloc(&d_b, sizeof(float) * out_f);
    cudaMalloc(&d_dW, sizeof(float) * in_f * out_f);
    cudaMalloc(&d_db, sizeof(float) * out_f);
    cudaMalloc(&d_Y, sizeof(float) * bs * out_f);
    cudaMalloc(&d_dX, sizeof(float) * bs * in_f);

    // FIX #3: Xavier/Glorot uniform initialization
    int threads = 256;
    int blocks = (in_f * out_f + threads - 1) / threads;
    xavier_init_kernel<<<blocks, threads>>>(d_W, in_f, out_f, (unsigned int)time(0) + in_f * 31 + out_f * 17);
    cudaMemset(d_b, 0, sizeof(float) * out_f);

    cudaMalloc(&d_m, sizeof(float) * in_f * out_f);
    cudaMalloc(&d_v, sizeof(float) * in_f * out_f);
    cudaMemset(d_m, 0, sizeof(float) * in_f * out_f);
    cudaMemset(d_v, 0, sizeof(float) * in_f * out_f);
}

LinearLayer::~LinearLayer() {
    cudaFree(d_W); cudaFree(d_b); cudaFree(d_dW); cudaFree(d_db);
    cudaFree(d_Y); cudaFree(d_dX); cudaFree(d_m); cudaFree(d_v);
}

float* LinearLayer::forward(cublasHandle_t h, void* inp, int act) {
    d_X_cache = (float*)inp;
    const float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                out_features, batch_features, in_features,
                &alpha, d_W, out_features, d_X_cache, in_features,
                &beta, d_Y, out_features);

    int threads = 256;
    int blocks_bias = (batch_features * out_features + threads - 1) / threads;
    add_bias_kernel<<<blocks_bias, threads>>>(d_Y, d_b, batch_features, out_features);

    // FIX #1: Actually APPLY the activation!
    if (act == ACTIVATION_RELU) {
        int total = batch_features * out_features;
        int blocks_act = (total + threads - 1) / threads;
        relu_forward_kernel<<<blocks_act, threads>>>(d_Y, total);
    } else if (act == ACTIVATION_GELU) {
        int total = batch_features * out_features;
        int blocks_act = (total + threads - 1) / threads;
        gelu_forward_kernel<<<blocks_act, threads>>>(d_Y, total);
    }

    return d_Y;
}

float* LinearLayer::backward(cublasHandle_t h, float* d_dY) {
    const float alpha = 1.0f, beta = 0.0f;

    // dX = dY * W^T
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                in_features, batch_features, out_features,
                &alpha, d_W, out_features, d_dY, out_features,
                &beta, d_dX, in_features);

    // dW = X^T * dY
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                out_features, in_features, batch_features,
                &alpha, d_dY, out_features, d_X_cache, in_features,
                &beta, d_dW, out_features);

    return d_dX;
}

void LinearLayer::step(float lr, int t) {
    int total = in_features * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    linear_adam_kernel<<<blocks, threads>>>(d_W, d_dW, d_m, d_v, lr, t, total);
}
