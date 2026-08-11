#pragma once
#include "Layer.cuh"
class LinearLayer : public Layer {
  float *d_W, *d_b, *d_dW, *d_db, *d_X_cache, *d_Y, *d_dX, *d_m, *d_v;
public:
  LinearLayer(int bs, int in_f, int out_f);
  ~LinearLayer();
  float* forward(cublasHandle_t h, void* inp, int act) override;
  float* backward(cublasHandle_t h, float* dY) override;
  void step(float lr, int t) override;
};
