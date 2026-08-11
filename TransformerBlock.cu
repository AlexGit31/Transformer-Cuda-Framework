#include "TransformerBlock.cuh"

__global__ void add_tensors_kernel(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] + B[idx];
}

TransformerBlock::TransformerBlock(int bs, int cs, int ed) : Layer(bs, cs, ed) {
    batch_size = bs; context_size = cs; embedding_dim = ed;
    norm1 = new RMSNormLayer(bs, cs, ed);
    attn  = new AttentionLayer(bs, cs, ed);
    norm2 = new RMSNormLayer(bs, cs, ed);
    ffn   = new FeedForwardLayer(bs, cs, ed, 4);
    int total = bs * cs * ed;
    cudaMalloc(&d_res1, sizeof(float) * total);
    cudaMalloc(&d_out,  sizeof(float) * total);
    cudaMalloc(&d_dRes1,sizeof(float) * total);
    cudaMalloc(&d_dX,   sizeof(float) * total);
}
TransformerBlock::~TransformerBlock() {
    delete norm1; delete attn; delete norm2; delete ffn;
    cudaFree(d_res1); cudaFree(d_out); cudaFree(d_dRes1); cudaFree(d_dX);
}

float* TransformerBlock::forward(cublasHandle_t h, void* inp, int act) {
    float* X = (float*)inp;
    int total = batch_size * context_size * embedding_dim;
    int t = 256;
    int b = (total + t - 1) / t;

    float* n1 = norm1->forward(h, X, ACTIVATION_NONE);
    float* a  = attn->forward(h, n1, ACTIVATION_NONE);
    add_tensors_kernel<<<b, t>>>(X, a, d_res1, total);
    cudaDeviceSynchronize();

    float* n2 = norm2->forward(h, d_res1, ACTIVATION_NONE);
    float* f  = ffn->forward(h, n2, ACTIVATION_NONE);
    add_tensors_kernel<<<b, t>>>(d_res1, f, d_out, total);
    cudaDeviceSynchronize();

    return d_out;
}

float* TransformerBlock::backward(cublasHandle_t h, float* dY) {
    int total = batch_size * context_size * embedding_dim;
    int t = 256;
    int b = (total + t - 1) / t;

    float* dF = ffn->backward(h, dY);
    float* dN2 = norm2->backward(h, dF);
    add_tensors_kernel<<<b, t>>>(dY, dN2, d_dRes1, total);
    cudaDeviceSynchronize();

    float* dA = attn->backward(h, d_dRes1);
    float* dN1 = norm1->backward(h, dA);
    add_tensors_kernel<<<b, t>>>(d_dRes1, dN1, d_dX, total);
    cudaDeviceSynchronize();

    return d_dX;
}

void TransformerBlock::step(float lr, int t) {
    norm1->step(lr, t); attn->step(lr, t);
    norm2->step(lr, t); ffn->step(lr, t);
}
