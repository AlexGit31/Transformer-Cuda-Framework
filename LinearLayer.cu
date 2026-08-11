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


// ─── Bias gradient kernel ──────────────────────────────────────────────
__global__ void bias_grad_kernel(float* d_db, float* d_dY, int batch_f, int out_f) {
    int tid = threadIdx.x;
    extern __shared__ float shared_sum[];
    shared_sum[tid] = 0.0f;
    for (int b = 0; b < batch_f; b++) {
        shared_sum[tid] += d_dY[b * out_f + tid];
    }
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_db[blockIdx.x * blockDim.x + tid] = shared_sum[0];
}


// ─── Bias Adam kernel (1D) ─────────────────────────────────────────────
__global__ void bias_adam_kernel(float* b, float* db, float* m, float* v, float lr, int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        m[idx] = b1 * m[idx] + (1.0f - b1) * db[idx];
        v[idx] = b2 * v[idx] + (1.0f - b2) * (db[idx] * db[idx]);
        float m_hat = m[idx] / (1.0f - powf(b1, (float)t));
        float v_hat = v[idx] / (1.0f - powf(b2, (float)t));
        b[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
        db[idx] = 0.0f;
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
    cudaMalloc(&d_pre_act, sizeof(float) * bs * out_f);
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

    cudaMalloc(&d_b_m, sizeof(float) * out_f);
    cudaMalloc(&d_b_v, sizeof(float) * out_f);
    cudaMemset(d_b_m, 0, sizeof(float) * out_f);
    cudaMemset(d_b_v, 0, sizeof(float) * out_f);
    last_activation = ACTIVATION_NONE;
}

LinearLayer::~LinearLayer() {
    cudaFree(d_W); cudaFree(d_b); cudaFree(d_dW); cudaFree(d_db);
    cudaFree(d_Y); cudaFree(d_pre_act); cudaFree(d_dX); cudaFree(d_m); cudaFree(d_v);
    cudaFree(d_b_m); cudaFree(d_b_v);
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

    // Save pre-activation for backward pass (only if activation will be applied)
    last_activation = act;
    if (act != ACTIVATION_NONE) {
        int total_act = batch_features * out_features;
        cudaMemcpy(d_pre_act, d_Y, sizeof(float) * total_act, cudaMemcpyDeviceToDevice);
        int blocks_act = (total_act + threads - 1) / threads;
        if (act == ACTIVATION_RELU) {
            relu_forward_kernel<<<blocks_act, threads>>>(d_Y, total_act);
        } else if (act == ACTIVATION_GELU) {
            gelu_forward_kernel<<<blocks_act, threads>>>(d_Y, total_act);
        }
    }

    return d_Y;
}

float* LinearLayer::backward(cublasHandle_t h, float* d_dY) {
    const float alpha = 1.0f, beta = 0.0f;
    int total = batch_features * out_features;
    int threads = 256;

    // FIX #8: Apply activation backward (GELU/ReLU derivative)
    if (last_activation == ACTIVATION_RELU) {
        int blocks_act = (total + threads - 1) / threads;
        relu_backward_kernel<<<blocks_act, threads>>>(d_dY, d_pre_act, total);
    } else if (last_activation == ACTIVATION_GELU) {
        int blocks_act = (total + threads - 1) / threads;
        gelu_backward_kernel<<<blocks_act, threads>>>(d_dY, d_pre_act, total);
    }

    // Compute bias gradient: d_db = sum(d_dY over batch)
    cudaMemset(d_db, 0, sizeof(float) * out_features);
    int blocks_bias = (out_features + threads - 1) / threads;
    bias_grad_kernel<<<blocks_bias, threads, threads * sizeof(float)>>>(d_db, d_dY, batch_features, out_features);

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

    // FIX #9: Also update bias with Adam
    int blocks_b = (out_features + threads - 1) / threads;
    bias_adam_kernel<<<blocks_b, threads>>>(d_b, d_db, d_b_m, d_b_v, lr, t, out_features);
}
