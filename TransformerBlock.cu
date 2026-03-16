#include "TransformerBlock.cuh"

// Le Kernel qui sauve l'IA : Addition élément par élément (C = A + B)
__global__ void add_tensors_kernel(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

TransformerBlock::TransformerBlock(int batch_size, int context_size, int embedding_dim)
    : Layer(batch_size, context_size, embedding_dim) {
    
    this->batch_size = batch_size;
    this->context_size = context_size;
    this->embedding_dim = embedding_dim;

    norm1 = new RMSNormLayer(batch_size, context_size, embedding_dim);
    attn  = new AttentionLayer(batch_size, context_size, embedding_dim);
    norm2 = new RMSNormLayer(batch_size, context_size, embedding_dim);
    ffn   = new FeedForwardLayer(batch_size, context_size, embedding_dim, 4);

    int total_elements = batch_size * context_size * embedding_dim;
    
    // Allocation pour les câbles de contournement
    cudaMalloc(&d_res1, sizeof(float) * total_elements);
    cudaMalloc(&d_out,  sizeof(float) * total_elements);
    cudaMalloc(&d_dRes1,sizeof(float) * total_elements);
    cudaMalloc(&d_dX,   sizeof(float) * total_elements);
}

TransformerBlock::~TransformerBlock() {
    delete norm1; delete attn; delete norm2; delete ffn;
    cudaFree(d_res1); cudaFree(d_out);
    cudaFree(d_dRes1); cudaFree(d_dX);
}

float* TransformerBlock::forward(cublasHandle_t handle, void* d_input, int activation_type) {
    float* d_X = (float*)d_input;
    int total = batch_size * context_size * embedding_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // --- CHEMIN 1 : L'ATTENTION ---
    float* d_norm1 = norm1->forward(handle, d_X, ACTIVATION_NONE);
    float* d_attn  = attn->forward(handle, d_norm1, ACTIVATION_NONE);
    // CABLE DE SAUVETAGE 1 : On additionne l'entrée et la sortie !
    add_tensors_kernel<<<blocks, threads>>>(d_X, d_attn, d_res1, total);
    cudaDeviceSynchronize();

    // --- CHEMIN 2 : LE FEEDFORWARD ---
    float* d_norm2 = norm2->forward(handle, d_res1, ACTIVATION_NONE);
    float* d_ffn   = ffn->forward(handle, d_norm2, ACTIVATION_NONE);
    // CABLE DE SAUVETAGE 2 : On additionne le chemin 1 et le FFN !
    add_tensors_kernel<<<blocks, threads>>>(d_res1, d_ffn, d_out, total);
    cudaDeviceSynchronize();

    return d_out;
}

float* TransformerBlock::backward(cublasHandle_t handle, float* d_dY) {
    int total = batch_size * context_size * embedding_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // --- RETOUR CHEMIN 2 (FFN) ---
    float* d_dFFN = ffn->backward(handle, d_dY);
    float* d_dNorm2_out = norm2->backward(handle, d_dFFN);
    // Jonction du câble : On additionne le gradient direct (d_dY) et le gradient du FFN
    add_tensors_kernel<<<blocks, threads>>>(d_dY, d_dNorm2_out, d_dRes1, total);
    cudaDeviceSynchronize();

    // --- RETOUR CHEMIN 1 (Attention) ---
    float* d_dAttn = attn->backward(handle, d_dRes1);
    float* d_dNorm1_out = norm1->backward(handle, d_dAttn);
    // Jonction du câble final : On additionne le gradient du chemin 2 et de l'Attention
    add_tensors_kernel<<<blocks, threads>>>(d_dRes1, d_dNorm1_out, d_dX, total);
    cudaDeviceSynchronize();

    return d_dX;
}

void TransformerBlock::step(float learning_rate, int t) {
    norm1->step(learning_rate, t);
    attn->step(learning_rate, t);
    norm2->step(learning_rate, t);
    ffn->step(learning_rate, t);
}
