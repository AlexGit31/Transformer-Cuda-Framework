#pragma once
#include <cublas_v2.h>

#define ACTIVATION_NONE 0
#define ACTIVATION_RELU 1
#define ACTIVATION_GELU 2

class Layer {
  protected:
    int batch_features, in_features, out_features;
  public:
    Layer(int bs, int in_f, int out_f) : batch_features(bs), in_features(in_f), out_features(out_f) {}
    virtual ~Layer() {}
    virtual float* forward(cublasHandle_t h, void* inp, int act) = 0;
    virtual float* backward(cublasHandle_t h, float* dY) = 0;
    virtual void step(float lr, int t) = 0;
};
