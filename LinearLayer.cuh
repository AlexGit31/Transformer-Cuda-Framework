#pragma once
#include "Layer.cuh"
#include <cublas_v2.h>

class LinearLayer : public Layer {
  private:
    float* d_W;
    float* d_b;
    float* d_dW;
    float* d_db;
    
    float* d_X_cache; // Pour sauvegarder l'entrée pendant le forward
    float* d_Y;       // La sortie
    float* d_dX;      // Le gradient à renvoyer vers le bas

    // ADAM
    float* d_m; // Momentum
    float* d_v; // Vélocité

  public:
    LinearLayer(int batch_size, int in_feat, int out_feat);
    ~LinearLayer();

    // Les 3 fameuses méthodes imposées par Layer.cuh
    float* forward(cublasHandle_t handle, void* d_input, int activation_type) override; 
    float* backward(cublasHandle_t handle, float* d_dY) override;
    void step(float learning_rate,int t) override;
};
