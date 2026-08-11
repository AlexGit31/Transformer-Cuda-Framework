#include "FeedForwardLayer.cuh"

FeedForwardLayer::FeedForwardLayer(int bs, int cs, int ed, int exp) : Layer(bs, cs, ed) {
    batch_size = bs; context_size = cs; embedding_dim = ed;
    hidden_dim = ed * exp;
    int fb = bs * cs;
    fc1 = new LinearLayer(fb, ed, hidden_dim);
    fc2 = new LinearLayer(fb, hidden_dim, ed);
}
FeedForwardLayer::~FeedForwardLayer() { delete fc1; delete fc2; }

float* FeedForwardLayer::forward(cublasHandle_t h, void* inp, int act) {
    // FIX #5: Use GELU instead of ReLU (modern transformer standard)
    float* hidden = fc1->forward(h, (float*)inp, ACTIVATION_GELU);
    return fc2->forward(h, hidden, ACTIVATION_NONE);
}
float* FeedForwardLayer::backward(cublasHandle_t h, float* dY) {
    float* dHidden = fc2->backward(h, dY);
    return fc1->backward(h, dHidden);
}
void FeedForwardLayer::step(float lr, int t) {
    fc1->step(lr, t); fc2->step(lr, t);
}
