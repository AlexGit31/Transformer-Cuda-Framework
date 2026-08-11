#include "AttentionLayer.cuh"
#include <cmath>

__global__ void causal_mask_kernel(float* Scores, int cs, int bs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bs * cs * cs;
    if (idx < total) {
        int m_idx = idx % (cs * cs);
        int row = m_idx / cs, col = m_idx % cs;
        if (col > row) Scores[idx] = -1e9f;
    }
}

__global__ void scale_scores_kernel(float* Scores, float scale, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) Scores[idx] *= scale;
}

__global__ void softmax_forward_kernel(float* Scores, int cs, int bs) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    if (row < bs * cs && tid < cs) {
        int base = row * cs;
        float max_val = -1e9f;
        for (int i = 0; i < cs; i++) max_val = fmaxf(max_val, Scores[base + i]);
        float my_exp = expf(Scores[base + tid] - max_val);
        shared[tid] = my_exp;
        __syncthreads();
        float sum = 0.0f;
        for (int i = 0; i < cs; i++) sum += shared[i];
        Scores[base + tid] = my_exp / (sum + 1e-9f);
    }
}

__global__ void softmax_backward_kernel(float* dSoft, float* Soft, float* dScores, int cs) {
    int row = blockIdx.x, tid = threadIdx.x;
    if (tid < cs) {
        int base = row * cs;
        float sum = 0.0f;
        for (int i = 0; i < cs; i++) sum += dSoft[base + i] * Soft[base + i];
        int idx = base + tid;
        dScores[idx] = Soft[idx] * (dSoft[idx] - sum);
    }
}

__global__ void sum_gradients_kernel(float* dX, const float* dXq, const float* dXk, const float* dXv, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) dX[idx] = dXq[idx] + dXk[idx] + dXv[idx];
}

AttentionLayer::AttentionLayer(int bs, int cs, int ed) : Layer(bs, cs, ed) {
    batch_size = bs; context_size = cs; embedding_dim = ed;
    int fb = bs * cs;
    W_q = new LinearLayer(fb, ed, ed);
    W_k = new LinearLayer(fb, ed, ed);
    W_v = new LinearLayer(fb, ed, ed);
    W_o = new LinearLayer(fb, ed, ed);
    cudaMalloc(&d_Scores, sizeof(float) * bs * cs * cs);
    cudaMalloc(&d_AttentionOut, sizeof(float) * bs * cs * ed);
    int te = bs * cs * ed, ts = bs * cs * cs;
    cudaMalloc(&d_Q_grad, sizeof(float) * te);
    cudaMalloc(&d_K_grad, sizeof(float) * te);
    cudaMalloc(&d_V_grad, sizeof(float) * te);
    cudaMalloc(&d_dX, sizeof(float) * te);
    cudaMalloc(&d_dScores_softmax, sizeof(float) * ts);
    cudaMalloc(&d_dScores, sizeof(float) * ts);
}
AttentionLayer::~AttentionLayer() {
    delete W_q; delete W_k; delete W_v; delete W_o;
    cudaFree(d_Scores); cudaFree(d_AttentionOut);
    cudaFree(d_Q_grad); cudaFree(d_K_grad); cudaFree(d_V_grad);
    cudaFree(d_dX); cudaFree(d_dScores_softmax); cudaFree(d_dScores);
}

float* AttentionLayer::forward(cublasHandle_t h, void* inp, int act) {
    float* X = (float*)inp;
    const float alpha = 1.0f, beta = 0.0f;

    d_Q = W_q->forward(h, X, ACTIVATION_NONE);
    d_K = W_k->forward(h, X, ACTIVATION_NONE);
    d_V = W_v->forward(h, X, ACTIVATION_NONE);

    long long strideE = context_size * embedding_dim;
    long long strideS = context_size * context_size;

    // Scores = Q @ K^T
    cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N,
        context_size, context_size, embedding_dim,
        &alpha, d_K, embedding_dim, strideE, d_Q, embedding_dim, strideE,
        &beta, d_Scores, context_size, strideS, batch_size);

    int totalS = batch_size * context_size * context_size;
    int t256 = 256;
    int bS = (totalS + t256 - 1) / t256;
    scale_scores_kernel<<<bS, t256>>>(d_Scores, 1.0f/sqrtf((float)embedding_dim), totalS);
    causal_mask_kernel<<<bS, t256>>>(d_Scores, context_size, batch_size);
    cudaDeviceSynchronize();

    int bSoft = batch_size * context_size;
    softmax_forward_kernel<<<bSoft, context_size, context_size*sizeof(float)>>>(d_Scores, context_size, batch_size);
    cudaDeviceSynchronize();

    // Out = Scores @ V
    cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
        embedding_dim, context_size, context_size,
        &alpha, d_V, embedding_dim, strideE, d_Scores, context_size, strideS,
        &beta, d_AttentionOut, embedding_dim, strideE, batch_size);

    return W_o->forward(h, d_AttentionOut, ACTIVATION_NONE);
}

float* AttentionLayer::backward(cublasHandle_t h, float* dY) {
    const float alpha = 1.0f, beta = 0.0f;
    long long strideE = context_size * embedding_dim;
    long long strideS = context_size * context_size;

    float* dOut = W_o->backward(h, dY);

    // dV, dScores_softmax
    cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T,
        embedding_dim, context_size, context_size,
        &alpha, dOut, embedding_dim, strideE, d_Scores, context_size, strideS,
        &beta, d_V_grad, embedding_dim, strideE, batch_size);

    cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N,
        context_size, context_size, embedding_dim,
        &alpha, d_V, embedding_dim, strideE, dOut, embedding_dim, strideE,
        &beta, d_dScores_softmax, context_size, strideS, batch_size);

    // Softmax backward
    int bSoft = batch_size * context_size;
    softmax_backward_kernel<<<bSoft, context_size>>>(d_dScores_softmax, d_Scores, d_dScores, context_size);
    cudaDeviceSynchronize();

    // Scale backward
    int totalS = batch_size * context_size * context_size;
    int t256 = 256;
    int bS = (totalS + t256 - 1) / t256;
    scale_scores_kernel<<<bS, t256>>>(d_dScores, 1.0f/sqrtf((float)embedding_dim), totalS);

    // dQ, dK from dScores
    cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
        embedding_dim, context_size, context_size,
        &alpha, d_K, embedding_dim, strideE, d_dScores, context_size, strideS,
        &beta, d_Q_grad, embedding_dim, strideE, batch_size);

    cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T,
        embedding_dim, context_size, context_size,
        &alpha, d_Q, embedding_dim, strideE, d_dScores, context_size, strideS,
        &beta, d_K_grad, embedding_dim, strideE, batch_size);

    float* dXq = W_q->backward(h, d_Q_grad);
    float* dXk = W_k->backward(h, d_K_grad);
    float* dXv = W_v->backward(h, d_V_grad);

    int totalE = batch_size * context_size * embedding_dim;
    int bE = (totalE + t256 - 1) / t256;
    sum_gradients_kernel<<<bE, t256>>>(d_dX, dXq, dXk, dXv, totalE);
    cudaDeviceSynchronize();
    return d_dX;
}

void AttentionLayer::step(float lr, int t) {
    W_q->step(lr, t); W_k->step(lr, t); W_v->step(lr, t); W_o->step(lr, t);
}
