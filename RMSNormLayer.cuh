#pragma once
#include "Layer.cuh"

class RMSNormLayer : public Layer {
  private:
    float* d_gamma;
    float* d_dgamma;
    float* d_m_gamma, *d_v_gamma;  // Adam for learnable gamma
    
    // Les tampons pour le backward bridé
    float* d_inv_rms; // Sauvegarde de la division (l'échelle)
    float* d_dX;      // Le gradient de sortie corrigé
    
    int batch_size;
    int context_size;
    int embedding_dim;

    float* d_Y;
    float* d_X_cache;

  public:
    // Le constructeur exact que le .cu va chercher
    RMSNormLayer(int batch_size, int context_size, int embedding_dim);
    ~RMSNormLayer();

    float* forward(cublasHandle_t handle, void* d_input, int activation_type) override; 
    float* backward(cublasHandle_t handle, float* d_dY) override;
    void step(float learning_rate, int t) override;
};
