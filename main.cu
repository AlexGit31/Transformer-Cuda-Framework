#include <iostream>
#include <cublas_v2.h>
#include "GPTModel.cuh"
#include "DataLoader.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <ctime>

// ─── NaN detection ────────────────────────────────────────────────────
__global__ void check_nan_kernel(float* t, int size, const char* name) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isnan(t[idx]) || isinf(t[idx]))
            printf("ALERT: NaN/Inf in %s at idx %d\n", name, idx);
    }
}

// ─── Gradient clipping ────────────────────────────────────────────────
__global__ void clip_gradients_kernel(float* dY, float min_v, float max_v, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = dY[idx];
        if (v > max_v) v = max_v;
        if (v < min_v) v = min_v;
        if (isnan(v)) v = 0.0f;
        dY[idx] = v;
    }
}

// ─── Fused Softmax + Cross-Entropy backward ───────────────────────────
__global__ void cross_entropy_backward_kernel(float* logits, int* targets, float* dY,
                                               int vocab_size, int total_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_words) {
        int target = targets[idx];
        float max_val = -1e9f;
        for (int i = 0; i < vocab_size; i++)
            max_val = fmaxf(max_val, logits[idx * vocab_size + i]);

        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++)
            sum_exp += expf(logits[idx * vocab_size + i] - max_val);

        for (int i = 0; i < vocab_size; i++) {
            float prob = expf(logits[idx * vocab_size + i] - max_val) / (sum_exp + 1e-7f);
            dY[idx * vocab_size + i] = (prob - (i == target ? 1.0f : 0.0f)) / (float)total_words;
        }
    }
}

// ─── Text generation ──────────────────────────────────────────────────
void generate_text(cublasHandle_t handle, GPTModel* model, std::string prompt,
                   int length, char* i2c, int cs, int vs) {
    std::cout << "\nPrompt: \"" << prompt << "\"\nGenerated: " << prompt;

    std::vector<int> ctx;
    for (char c : prompt) {
        int tok = 0;
        for (int v = 0; v < vs; v++) { if (i2c[v] == c) { tok = v; break; } }
        ctx.push_back(tok);
    }

    int* d_X; cudaMalloc(&d_X, sizeof(int) * cs);
    float* h_logits = (float*)malloc(sizeof(float) * cs * vs);

    for (int i = 0; i < length; i++) {
        std::vector<int> win;
        int start = std::max(0, (int)ctx.size() - cs);
        for (int j = start; j < (int)ctx.size(); j++) win.push_back(ctx[j]);
        while ((int)win.size() < cs) win.push_back(0);

        cudaMemcpy(d_X, win.data(), sizeof(int) * cs, cudaMemcpyHostToDevice);
        float* d_logits = model->forward(handle, d_X, 0);
        cudaMemcpy(h_logits, d_logits, sizeof(float) * cs * vs, cudaMemcpyDeviceToHost);

        int off = (cs - 1) * vs;
        float max_l = -1e9f;
        for (int v = 0; v < vs; v++) max_l = std::max(max_l, h_logits[off + v]);

        float sum_exp = 0.0f;
        std::vector<float> probs(vs);
        for (int v = 0; v < vs; v++) {
            probs[v] = expf(h_logits[off + v] - max_l);
            sum_exp += probs[v];
        }

        float r = ((float)rand() / RAND_MAX) * sum_exp;
        float cum = 0.0f;
        int next = 0;
        for (int v = 0; v < vs; v++) { cum += probs[v]; if (r <= cum) { next = v; break; } }

        std::cout << i2c[next] << std::flush;
        ctx.push_back(next);
    }
    std::cout << std::endl;
    cudaFree(d_X); free(h_logits);
}

// ─── MAIN ─────────────────────────────────────────────────────────────
int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // ── Hyperparameters ───────────────────────────────────────────────
    int context_size = 32;
    int batch_size = 128;
    int embedding_dim = 128;
    int num_blocks = 4;
    float base_lr = 3e-4f;
    int total_iterations = 10000;
    int warmup_steps = 1000;
    int log_every = 50;
    int eval_every = 500;

    std::cout << "=== GPT-CUDA v2 (Fixed) ===\n";
    std::cout << "Context: " << context_size << ", Batch: " << batch_size
              << ", Emb: " << embedding_dim << ", Blocks: " << num_blocks << "\n";
    std::cout << "Base LR: " << base_lr << ", Warmup: " << warmup_steps
              << ", Iterations: " << total_iterations << "\n";

    // ── Data ──────────────────────────────────────────────────────────
    DataLoader dataloader("input.txt", batch_size, context_size);
    int vocab_size = dataloader.get_vocab_size();

    // ── Model ─────────────────────────────────────────────────────────
    GPTModel model(vocab_size, embedding_dim, batch_size, context_size, num_blocks);
    int total_words = batch_size * context_size;

    // ── GPU memory ────────────────────────────────────────────────────
    int *h_X = (int*)malloc(sizeof(int) * total_words);
    int *h_targets = (int*)malloc(sizeof(int) * total_words);
    int *d_X, *d_targets; float *d_dY;
    cudaMalloc(&d_X, sizeof(int) * total_words);
    cudaMalloc(&d_targets, sizeof(int) * total_words);
    cudaMalloc(&d_dY, sizeof(float) * total_words * vocab_size);

    // ── Logging ───────────────────────────────────────────────────────
    std::ofstream log_file("training_log.csv");
    log_file << "iteration,loss,lr\n";

    std::cout << "\n=== TRAINING ===\n";
    time_t start_time = time(0);

    for (int iter = 0; iter < total_iterations; iter++) {
        // FIX #4: Learning rate warmup (linear)
        float lr;
        if (iter < warmup_steps) {
            lr = base_lr * ((float)(iter + 1) / (float)warmup_steps);
        } else {
            // Cosine decay after warmup
            float progress = (float)(iter - warmup_steps) / (float)(total_iterations - warmup_steps);
            lr = base_lr * 0.5f * (1.0f + cosf(3.14159265f * progress));
        }

        dataloader.get_batch(h_X, h_targets);
        cudaMemcpy(d_X, h_X, sizeof(int) * total_words, cudaMemcpyHostToDevice);
        cudaMemcpy(d_targets, h_targets, sizeof(int) * total_words, cudaMemcpyHostToDevice);

        // Forward
        float* d_logits = model.forward(handle, d_X, 0);

        // Loss computation (every log_every steps)
        if (iter % log_every == 0) {
            float* h_logits = (float*)malloc(sizeof(float) * total_words * vocab_size);
            cudaMemcpy(h_logits, d_logits, sizeof(float) * total_words * vocab_size, cudaMemcpyDeviceToHost);

            float loss = 0.0f;
            for (int i = 0; i < total_words; i++) {
                int target = h_targets[i];
                float max_l = -1e9f;
                for (int v = 0; v < vocab_size; v++)
                    max_l = std::max(max_l, h_logits[i * vocab_size + v]);

                float sum_exp = 0.0f;
                for (int v = 0; v < vocab_size; v++)
                    sum_exp += expf(h_logits[i * vocab_size + v] - max_l);

                float prob = expf(h_logits[i * vocab_size + target] - max_l) / sum_exp;
                loss += -logf(prob + 1e-7f);
            }
            loss /= total_words;
            free(h_logits);

            time_t elapsed = time(0) - start_time;
            std::cout << "[" << iter << "/" << total_iterations << "] loss="
                      << loss << " lr=" << lr << " time=" << elapsed << "s\n";
            log_file << iter << "," << loss << "," << lr << "\n";
        }

        // Backward
        int threads = 256;
        int blocks_ce = (total_words + threads - 1) / threads;
        cross_entropy_backward_kernel<<<blocks_ce, threads>>>(
            d_logits, d_targets, d_dY, vocab_size, total_words);

        // Gradient clipping
        int total_grad = total_words * vocab_size;
        int blocks_clip = (total_grad + threads - 1) / threads;
        clip_gradients_kernel<<<blocks_clip, threads>>>(d_dY, -1.0f, 1.0f, total_grad);
        cudaDeviceSynchronize();

        model.backward(handle, d_dY);
        model.step(lr, iter + 1);  // iter+1 for Adam bias correction

        // NaN check (every 500 steps)
        if (iter % 500 == 0) {
            int total_logits = total_words * vocab_size;
            check_nan_kernel<<<(total_logits + 255)/256, 256>>>(d_logits, total_logits, "logits");
            cudaDeviceSynchronize();
        }
    }

    log_file.close();
    std::cout << "\n=== TRAINING COMPLETE ===\n";

    // ── Generation ────────────────────────────────────────────────────
    std::map<int, char> i2c_map = dataloader.get_i2c();
    char* i2c = (char*)malloc(sizeof(char) * vocab_size);
    for (int i = 0; i < vocab_size; i++) i2c[i] = i2c_map[i];

    std::cout << "\n=== TEXT GENERATION ===\n";
    generate_text(handle, &model, "First Citizen:", 100, i2c, context_size, vocab_size);
    generate_text(handle, &model, "ROMEO:", 100, i2c, context_size, vocab_size);
    generate_text(handle, &model, "The king", 100, i2c, context_size, vocab_size);

    // Cleanup
    free(h_X); free(h_targets); free(i2c);
    cudaFree(d_X); cudaFree(d_targets); cudaFree(d_dY);
    cublasDestroy(handle);

    return 0;
}
