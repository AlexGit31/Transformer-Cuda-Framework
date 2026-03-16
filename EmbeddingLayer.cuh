#pragma once
#include "Layer.cuh"

class EmbeddingLayer : public Layer {
  private:
    // Poids et gradients
    float* d_W;  // Le dictionnaire [vocab_size * embedding_dim]
    float* d_dW; // Le gradient des poids

    // Encodage Positionnel (Constant)
    float* d_PE; // La matrice des ondes [context_size * embedding_dim]
    
    // Entrées / Sorties
    int* d_X;    // L'entrée (Tableau d'entiers) [batch_size * context_size]
    float* d_Y;  // La sortie [batch_size * context_size * embedding_dim]
    
    // Les dimensions
    int vocab_size;
    int embedding_dim;
    int batch_size;
    int context_size;

    //ADAM
    float* d_m;
    float* d_v;
    
  public:
    EmbeddingLayer(int vocab_size, int embedding_dim, int batch_size, int context_size);
    ~EmbeddingLayer();

    float* forward(cublasHandle_t handle, void* d_input, int activation_type) override; 
    float* backward(cublasHandle_t handle, float* d_dY) override;
    void step(float learning_rate,int t) override;
};
