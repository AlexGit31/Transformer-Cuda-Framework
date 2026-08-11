#pragma once
#include "Layer.cuh"
#include "LinearLayer.cuh"
class FeedForwardLayer : public Layer {
  LinearLayer *fc1, *fc2;
  int batch_size, context_size, embedding_dim, hidden_dim;
public:
  FeedForwardLayer(int bs, int cs, int ed, int exp=4);
  ~FeedForwardLayer();
  float* forward(cublasHandle_t h, void* inp, int act) override;
  float* backward(cublasHandle_t h, float* dY) override;
  void step(float lr, int t) override;
};
