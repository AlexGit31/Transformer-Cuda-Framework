#pragma once
#include "Layer.cuh"
#include "AttentionLayer.cuh"
#include "FeedForwardLayer.cuh"
#include "RMSNormLayer.cuh"
class TransformerBlock : public Layer {
  RMSNormLayer *norm1, *norm2;
  AttentionLayer *attn;
  FeedForwardLayer *ffn;
  float *d_res1, *d_out, *d_dRes1, *d_dX;
  int batch_size, context_size, embedding_dim;
public:
  TransformerBlock(int bs, int cs, int ed);
  ~TransformerBlock();
  float* forward(cublasHandle_t h, void* inp, int act) override;
  float* backward(cublasHandle_t h, float* dY) override;
  void step(float lr, int t) override;
};
