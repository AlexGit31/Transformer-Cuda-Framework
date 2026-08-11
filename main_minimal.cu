/**
 * GPT-CUDA MINIMAL: Bigram model (embedding + linear head, NO transformer blocks)
 * Diagnostic test: if this learns, the bug is in the transformer blocks.
 * If this doesn't learn, the bug is in the training loop.
 */
#include <iostream>
#include <cublas_v2.h>
#include "EmbeddingLayer.cuh"
#include "LinearLayer.cuh"
#include "DataLoader.h"
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <ctime>

__global__ void cross_entropy_backward_kernel(float* logits, int* targets, float* dY,
                                               int vs, int tw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tw) {
        int tgt = targets[idx];
        float mx = -1e9f;
        for (int i = 0; i < vs; i++) mx = fmaxf(mx, logits[idx * vs + i]);
        float sum = 0.0f;
        for (int i = 0; i < vs; i++) sum += expf(logits[idx * vs + i] - mx);
        for (int i = 0; i < vs; i++) {
            float p = expf(logits[idx * vs + i] - mx) / (sum + 1e-7f);
            dY[idx * vs + i] = (p - (i == tgt ? 1.0f : 0.0f)) / (float)tw;
        }
    }
}

__global__ void clip_gradients_kernel(float* dY, float mn, float mx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = dY[idx];
        if (v > mx) v = mx; if (v < mn) v = mn;
        if (isnan(v)) v = 0.0f;
        dY[idx] = v;
    }
}

int main() {
    cublasHandle_t h; cublasCreate(&h);

    int cs = 64, bs = 64, ed = 192;
    float base_lr = 3e-3f;
    int total_iter = 5000, warmup = 1000, log_every = 50;

    DataLoader dl("input.txt", bs, cs);
    int vs = dl.get_vocab_size();
    int tw = bs * cs;

    std::cout << "=== GPT-CUDA MINIMAL (Bigram: Embedding + LM Head only) ===\n";
    std::cout << "Vocab: " << vs << ", Emb: " << ed << ", Tokens/batch: " << tw << "\n";

    // Minimal model: embedding + linear head (no transformer blocks)
    EmbeddingLayer emb(vs, ed, bs, cs);
    LinearLayer head(bs * cs, ed, vs);

    int *hX = (int*)malloc(sizeof(int)*tw), *hT = (int*)malloc(sizeof(int)*tw);
    int *dX, *dT; float *d_dY;
    cudaMalloc(&dX, sizeof(int)*tw);
    cudaMalloc(&dT, sizeof(int)*tw);
    cudaMalloc(&d_dY, sizeof(float)*tw*vs);

    std::ofstream log("training_log_minimal.csv");
    log << "iteration,loss,lr\n";
    time_t t0 = time(0);

    for (int iter = 0; iter < total_iter; iter++) {
        float lr;
        if (iter < warmup)
            lr = base_lr * ((float)(iter+1)/(float)warmup);
        else {
            float p = (float)(iter-warmup)/(float)(total_iter-warmup);
            lr = base_lr * (0.1f + 0.45f*(1.0f+cosf(3.14159265f*p)));
        }

        dl.get_batch(hX, hT);
        cudaMemcpy(dX, hX, sizeof(int)*tw, cudaMemcpyHostToDevice);
        cudaMemcpy(dT, hT, sizeof(int)*tw, cudaMemcpyHostToDevice);

        float* d_emb = emb.forward(h, dX, 0);
        float* d_logits = head.forward(h, d_emb, 0);

        if (iter % log_every == 0) {
            float* h_logits = (float*)malloc(sizeof(float)*tw*vs);
            cudaMemcpy(h_logits, d_logits, sizeof(float)*tw*vs, cudaMemcpyDeviceToHost);
            float loss = 0.0f;
            for (int i = 0; i < tw; i++) {
                int tgt = hT[i];
                float mx = -1e9f;
                for (int v = 0; v < vs; v++) mx = std::max(mx, h_logits[i*vs+v]);
                float sum = 0.0f;
                for (int v = 0; v < vs; v++) sum += expf(h_logits[i*vs+v] - mx);
                loss += -logf(expf(h_logits[i*vs+tgt]-mx)/sum + 1e-7f);
            }
            loss /= tw;
            free(h_logits);
            std::cout << "[" << iter << "/" << total_iter << "] loss=" << loss
                      << " lr=" << lr << " time=" << (time(0)-t0) << "s\n";
            log << iter << "," << loss << "," << lr << "\n";
        }

        int th = 256;
        int bce = (tw + th - 1) / th;
        cross_entropy_backward_kernel<<<bce, th>>>(d_logits, dT, d_dY, vs, tw);
        int tgrad = tw * vs;
        int bclip = (tgrad + th - 1) / th;
        clip_gradients_kernel<<<bclip, th>>>(d_dY, -5.0f, 5.0f, tgrad);
        cudaDeviceSynchronize();

        float* d_grad = head.backward(h, d_dY);
        emb.backward(h, d_grad);
        head.step(lr, iter+1);
        emb.step(lr, iter+1);
    }

    log.close();
    std::cout << "\n=== DONE ===\n";

    free(hX); free(hT);
    cudaFree(dX); cudaFree(dT); cudaFree(d_dY);
    cublasDestroy(h);
    return 0;
}
