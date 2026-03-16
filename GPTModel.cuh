#pragma once
#include "Layer.cuh"
#include "EmbeddingLayer.cuh"
#include "TransformerBlock.cuh"
#include "RMSNormLayer.cuh"
#include "LinearLayer.cuh"
#include <vector>

class GPTModel : public Layer {
  private:
    // La liste complète de nos composants
    EmbeddingLayer* embedding;
    
    // Un tableau dynamique pour stocker nos N blocs Transformer
    std::vector<TransformerBlock*> blocks; 
    
    RMSNormLayer* final_norm;
    LinearLayer* lm_head; // La tête de prédiction

    // L'architecture
    int vocab_size;
    int embedding_dim;
    int batch_size;
    int context_size;
    int num_blocks; // Combien de couches d'Attention on empile ?

  public:
    GPTModel(int vocab_size, int embedding_dim, int batch_size, int context_size, int num_blocks);
    ~GPTModel();

    float* forward(cublasHandle_t handle, void* d_input, int activation_type) override; 
    float* backward(cublasHandle_t handle, float* d_dY) override;
    void step(float learning_rate,int t) override;
};
