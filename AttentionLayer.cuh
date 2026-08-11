#pragma once
#include "Layer.cuh"
#include "LinearLayer.cuh"
class AttentionLayer : public Layer {
  LinearLayer *W_q, *W_k, *W_v, *W_o;
  float *d_Q, *d_K, *d_V, *d_Scores, *d_AttentionOut;
  float *d_Q_grad, *d_K_grad, *d_V_grad, *d_dScores_softmax, *d_dScores, *d_dX;
  int batch_size, context_size, embedding_dim;
public:
  AttentionLayer(int bs, int cs, int ed);
  ~AttentionLayer();
  float* forward(cublasHandle_t h, void* inp, int act) override;
  float* backward(cublasHandle_t h, float* dY) override;
  void step(float lr, int t) override;
};
