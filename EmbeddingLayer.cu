#include "EmbeddingLayer.cuh"
#include <iostream>
#include <cmath>
#include <cstdlib>

// =========================================================================
// KERNELS CUDA
// =========================================================================



__global__ void embedding_adam_kernel(float* W, float* dW, float* m, float* v, float lr, int t, int size) {
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
        dW[idx] = 0.0f; 
    }
}


// 1. Kernel Forward : Copie du Dictionnaire + Addition de l'Onde Spatiale (Kernel Fusion)
__global__ void embedding_forward_kernel(int* d_X, float* d_W, float* d_PE, float* d_Y, int embedding_dim, int context_size, int total_elements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < total_elements) {
        int embed_idx = idx % embedding_dim;       // Quelle colonne (0 à 255) ?
        int word_pos_global = idx / embedding_dim; // Quel mot dans tout le batch ?
        
        // Position relative de 0 à context_size-1 (pour savoir quelle onde utiliser)
        int seq_pos = word_pos_global % context_size; 
        
        int vocab_id = d_X[word_pos_global];       // L'ID du mot (ex: 45)
        
        // La Fusion : Dictionnaire + Position
        d_Y[idx] = d_W[vocab_id * embedding_dim + embed_idx] + d_PE[seq_pos * embedding_dim + embed_idx];
    }
}

// 2. Kernel Backward : Accumulation des Gradients
__global__ void backward_embedding_kernel(int* d_X, float* d_dY, float* d_dW, int embedding_dim, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int embed_idx = idx % embedding_dim;
        int word_pos  = idx / embedding_dim;
        int vocab_id  = d_X[word_pos];
        
        // atomicAdd pour éviter les collisions si un mot apparait plusieurs fois
        atomicAdd(&d_dW[vocab_id * embedding_dim + embed_idx], d_dY[idx]);
    }
}

// 3. Kernel SGD : Mise à jour des poids du Dictionnaire
__global__ void sgd_update_emb(float* param, const float* grad, float lr, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        param[idx] = param[idx] - (lr * grad[idx]);
    }
}

// =========================================================================
// MÉTHODES DE LA CLASSE
// =========================================================================

EmbeddingLayer::EmbeddingLayer(int vocab_size, int embedding_dim, int batch_size, int context_size) 
    : Layer(batch_size, context_size, embedding_dim) { // Appel au parent !
    
    this->vocab_size = vocab_size;
    this->embedding_dim = embedding_dim;
    this->batch_size = batch_size;
    this->context_size = context_size;

    // 1. Allocations VRAM
    cudaMalloc(&d_W, sizeof(float) * vocab_size * embedding_dim);
    cudaMalloc(&d_dW, sizeof(float) * vocab_size * embedding_dim);
    cudaMalloc(&d_Y, sizeof(float) * batch_size * context_size * embedding_dim);
    cudaMalloc(&d_PE, sizeof(float) * context_size * embedding_dim);

    // 2. Initialisation du Dictionnaire W (CPU -> GPU)
    float *h_W = (float*)malloc(sizeof(float) * vocab_size * embedding_dim);
    for(int i = 0 ; i < vocab_size ; i++) {
        for(int j = 0 ; j < embedding_dim ; j++) {
            h_W[i * embedding_dim + j] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.05f;;
        }
    }
    cudaMemcpy(d_W, h_W, sizeof(float) * vocab_size * embedding_dim, cudaMemcpyHostToDevice);
    free(h_W);

    // 3. Calcul de l'Encodage Positionnel (Sinus/Cosinus) sur CPU
    float *h_PE = (float*)malloc(sizeof(float) * context_size * embedding_dim);
    for(int pos = 0; pos < context_size; pos++) {
        for(int i = 0; i < embedding_dim; i+=2) {
            // La formule du papier "Attention Is All You Need"
            float div_term = pow(10000.0f, (float)i / embedding_dim);
            
            // Les dimensions paires reçoivent le sinus
            h_PE[pos * embedding_dim + i] = sin(pos / div_term);
            
            // Les dimensions impaires reçoivent le cosinus
            if(i + 1 < embedding_dim) {
                h_PE[pos * embedding_dim + i + 1] = cos(pos / div_term);
            }
        }
    }
    // Envoi de l'Horloge sur le GPU une bonne fois pour toutes !
    cudaMemcpy(d_PE, h_PE, sizeof(float) * context_size * embedding_dim, cudaMemcpyHostToDevice);
    free(h_PE);
    cudaMalloc(&d_m, sizeof(float) * vocab_size * embedding_dim);
    cudaMalloc(&d_v, sizeof(float) * vocab_size * embedding_dim);
    cudaMemset(d_m, 0, sizeof(float) * vocab_size * embedding_dim);
    cudaMemset(d_v, 0, sizeof(float) * vocab_size * embedding_dim);
}

EmbeddingLayer::~EmbeddingLayer() {
    cudaFree(d_W);
    cudaFree(d_dW);
    cudaFree(d_Y);
    cudaFree(d_PE);
    cudaFree(d_m);
    cudaFree(d_v);
    // Note : On ne free pas d_X car il appartient au DataLoader
}

float* EmbeddingLayer::forward(cublasHandle_t handle, void* d_input, int activation_type) {
    d_X = (int*) d_input; 
    
    // 1. On calcule le nombre total de mots dans tout le batch
    int total_words = batch_size * context_size;
    
    // 2. La grille CUDA s'adapte au nombre de mots (1 thread = 1 mot)
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_words + threadsPerBlock - 1) / threadsPerBlock;
    
    // 3. Appel du kernel avec l'ORDRE EXACT des paramètres
    embedding_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_X, d_W, d_PE, d_Y, total_words, context_size, embedding_dim
    );
    
    cudaDeviceSynchronize();
    return d_Y;
}

float* EmbeddingLayer::backward(cublasHandle_t handle, float* d_dY) {
    int total_elements = batch_size * context_size * embedding_dim;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    // TRÈS IMPORTANT : Remettre le gradient à zéro avant l'accumulation !
    cudaMemset(d_dW, 0, vocab_size * embedding_dim * sizeof(float));

    backward_embedding_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_X, d_dY, d_dW, embedding_dim, total_elements
    );
    
    cudaDeviceSynchronize();
    return nullptr; // Le stop absolu.
}

void EmbeddingLayer::step(float learning_rate, int t) {
    int total_elements = vocab_size * embedding_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    embedding_adam_kernel<<<blocks, threads>>>(d_W, d_dW, d_m, d_v, learning_rate, t, total_elements);
}
