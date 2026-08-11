#pragma once
#include "Layer.cuh"
#include "EmbeddingLayer.cuh"
#include "TransformerBlock.cuh"
#include "RMSNormLayer.cuh"
#include "LinearLayer.cuh"
#include <vector>
class GPTModel : public Layer {
  EmbeddingLayer* emb; std::vector<TransformerBlock*> blocks;
  RMSNormLayer* fnorm; LinearLayer* lm_head;
  int vs, ed, bs, cs, nb;
public:
  GPTModel(int vs, int ed, int bs, int cs, int nb);
  ~GPTModel();
  float* forward(cublasHandle_t h, void* inp, int act) override;
  float* backward(cublasHandle_t h, float* dY) override;
  void step(float lr, int t) override;
};
