#pragma once
#include "Layer.cuh"
#include "AttentionLayer.cuh"
#include "FeedForwardLayer.cuh"
#include "RMSNormLayer.cuh"

class TransformerBlock : public Layer {
private:
    RMSNormLayer* norm1;
    AttentionLayer* attn;
    RMSNormLayer* norm2;
    FeedForwardLayer* ffn;

    // NOUVEAU : Mémoire pour les câbles de contournement (Residuals)
    float* d_res1; // Stocke : Entrée + Attention
    float* d_out;  // Stocke : res1 + FeedForward

    // NOUVEAU : Mémoire pour remonter les gradients
    float* d_dRes1; 
    float* d_dX;    // Le gradient final à renvoyer en dessous

    int batch_size;
    int context_size;
    int embedding_dim;

public:
    TransformerBlock(int batch_size, int context_size, int embedding_dim);
    ~TransformerBlock();

    float* forward(cublasHandle_t handle, void* d_input, int activation_type) override;
    float* backward(cublasHandle_t handle, float* d_dY) override;
    void step(float learning_rate, int t) override;
};
