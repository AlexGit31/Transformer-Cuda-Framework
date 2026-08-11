#include "GPTModel.cuh"

GPTModel::GPTModel(int vs, int ed, int bs, int cs, int nb) : Layer(bs, cs, ed) {
    this->vs = vs; this->ed = ed; this->bs = bs; this->cs = cs; this->nb = nb;
    emb = new EmbeddingLayer(vs, ed, bs, cs);
    for (int i = 0; i < nb; i++) blocks.push_back(new TransformerBlock(bs, cs, ed));
    fnorm = new RMSNormLayer(bs, cs, ed);
    lm_head = new LinearLayer(bs * cs, ed, vs);
}
GPTModel::~GPTModel() {
    delete emb; for (auto* b : blocks) delete b; delete fnorm; delete lm_head;
}

float* GPTModel::forward(cublasHandle_t h, void* inp, int act) {
    float* out = emb->forward(h, inp, ACTIVATION_NONE);
    for (auto* b : blocks) out = b->forward(h, out, ACTIVATION_NONE);
    out = fnorm->forward(h, out, ACTIVATION_NONE);
    return lm_head->forward(h, out, ACTIVATION_NONE);
}

float* GPTModel::backward(cublasHandle_t h, float* dY) {
    float* grad = lm_head->backward(h, dY);
    grad = fnorm->backward(h, grad);
    for (int i = nb - 1; i >= 0; i--) grad = blocks[i]->backward(h, grad);
    emb->backward(h, grad);
    return nullptr;
}

void GPTModel::step(float lr, int t) {
    emb->step(lr, t);
    for (auto* b : blocks) b->step(lr, t);
    fnorm->step(lr, t);
    lm_head->step(lr, t);
}
