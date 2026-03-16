#pragma once
#include "Layer.cuh"
#include "LinearLayer.cuh"

class FeedForwardLayer : public Layer {
  private:
    // Nos deux sous-couches
    LinearLayer* fc1; // L'expansion (x4)
    LinearLayer* fc2; // La contraction (x1)

    // Dimensions
    int batch_size;
    int context_size;
    int embedding_dim;
    int hidden_dim; // Généralement 4 * embedding_dim

  public:
    FeedForwardLayer(int batch_size, int context_size, int embedding_dim, int expansion_factor = 4);
    ~FeedForwardLayer();

    float* forward(cublasHandle_t handle, void* d_input, int activation_type) override; 
    float* backward(cublasHandle_t handle, float* d_dY) override;
    void step(float learning_rate,int t) override;
};
