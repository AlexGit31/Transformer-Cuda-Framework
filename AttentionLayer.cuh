#pragma once
#include "Layer.cuh"
#include "LinearLayer.cuh" // On importe notre propre outil !

class AttentionLayer : public Layer {
  private:
    // 1. Nos sous-couches (Les générateurs de Q, K et V)
    LinearLayer* W_q;
    LinearLayer* W_k;
    LinearLayer* W_v;
    
    // (Optionnel mais standard) Une dernière couche linéaire pour mélanger 
    // le résultat final avant de le passer au MLP
    LinearLayer* W_o; 

    // 2. Nos espaces mémoires intermédiaires sur le GPU
    float* d_Q;
    float* d_K;
    float* d_V;
    float* d_Scores;       // Pour stocker (Q * K^T)
    float* d_AttentionOut; // Pour stocker (Scores * V)
    
    // 3. Les dimensions
    int batch_size;
    int context_size;
    int embedding_dim;
    // NOUVEAU : Les tampons (buffers) pour le Backward !
    float* d_Q_grad;
    float* d_K_grad;
    float* d_V_grad;
    float* d_dScores_softmax;
    float* d_dScores;
    float* d_dX; // Le gradient final qui ressort de l'Attention

  public:
    AttentionLayer(int batch_size, int context_size, int embedding_dim);
    ~AttentionLayer();

    float* forward(cublasHandle_t handle, void* d_input, int activation_type) override; 
    float* backward(cublasHandle_t handle, float* d_dY) override;
    void step(float learning_rate,int t) override;
};
