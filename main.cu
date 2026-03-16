#include <iostream>
#include <cublas_v2.h>
#include "GPTModel.cuh"
#include "DataLoader.h"
#include <vector>
#include <string>

__global__ void check_nan_kernel(float* tensor, int size, const char* nom_couche) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isnan(tensor[idx]) || isinf(tensor[idx])) {
            printf("🚨 ALERTE : NaN ou Inf detecte dans la couche : %s (Index %d)\n", nom_couche, idx);
        }
    }
}

// Kernel de sécurité : Empêche les gradients d'exploser (Gradient Clipping)
__global__ void clip_gradients_kernel(float* d_dY, float min_val, float max_val, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = d_dY[idx];
        if (val > max_val) val = max_val;
        if (val < min_val) val = min_val;
        // La protection suprême contre les NaN générés plus haut :
        if (isnan(val)) val = 0.0f; 
        d_dY[idx] = val;
    }
}
// Fonction CPU pour générer du texte avec le modèle entraîné
// --- NOUVELLE FONCTION GENERATE_TEXT ---
void generate_text(cublasHandle_t handle, GPTModel* model, std::string prompt, int length_to_generate, char* int_to_char, int context_size, int vocab_size) {
    std::cout << "\nAmorce : \"" << prompt << "\"" << std::endl;
    std::cout << "Résultat : " << prompt;

    std::vector<int> current_context;
    
    // 1. CORRECTION : On utilise le VRAI dictionnaire pour traduire le prompt !
    for (char c : prompt) {
        int token = 0; // Token par défaut si caractère inconnu
        for(int v = 0; v < vocab_size; v++) {
            if (int_to_char[v] == c) {
                token = v;
                break;
            }
        }
        current_context.push_back(token);
    }

    int* d_X_gen;
    cudaMalloc(&d_X_gen, sizeof(int) * context_size);
    float* h_logits = (float*)malloc(sizeof(float) * context_size * vocab_size);

    for (int i = 0; i < length_to_generate; i++) {
        std::vector<int> input_window;
        int start_idx = std::max(0, (int)current_context.size() - context_size);
        for (int j = start_idx; j < current_context.size(); j++) {
            input_window.push_back(current_context[j]);
        }
        while(input_window.size() < context_size) input_window.push_back(0); 

        cudaMemcpy(d_X_gen, input_window.data(), sizeof(int) * context_size, cudaMemcpyHostToDevice);

        // Température ajoutée implicitement en ne modifiant pas les logits bruts
        float* d_logits_out = model->forward(handle, d_X_gen, 0);
        cudaMemcpy(h_logits, d_logits_out, sizeof(float) * context_size * vocab_size, cudaMemcpyDeviceToHost);
        
        int last_word_offset = (context_size - 1) * vocab_size;

        float max_l = -1e9f;
        for(int v=0; v<vocab_size; v++) max_l = std::max(max_l, h_logits[last_word_offset + v]);
        
        float sum_exp = 0.0f;
        std::vector<float> probs(vocab_size);
        for(int v=0; v<vocab_size; v++) {
            probs[v] = expf(h_logits[last_word_offset + v] - max_l);
            sum_exp += probs[v];
        }
        
        float r = ((float)rand() / RAND_MAX) * sum_exp;
        float cumulative = 0.0f;
        int next_token = 0;
        
        for(int v=0; v<vocab_size; v++) {
            cumulative += probs[v];
            if (r <= cumulative) {
                next_token = v;
                break;
            }
        }

        std::cout << int_to_char[next_token] << std::flush; // On affiche instantanément
        current_context.push_back(next_token);
    }
    
    std::cout << std::endl;
    cudaFree(d_X_gen);
    free(h_logits);
}

// Kernel magique : Fused Softmax + Cross Entropy Backward
// Il transforme les Logits bruts en gradients d_dY prêts à être rétropropagés.
__global__ void cross_entropy_backward_kernel(float* d_logits, int* d_targets, float* d_dY, int vocab_size, int total_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Un thread = un mot du batch
    
    if (idx < total_words) {
        int target_class = d_targets[idx]; // Le vrai mot attendu
        
        // 1. Softmax local (très simplifié ici pour l'exemple)
        float max_val = -1e9f;
        for (int i = 0; i < vocab_size; i++) {
            max_val = fmaxf(max_val, d_logits[idx * vocab_size + i]);
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            sum_exp += expf(d_logits[idx * vocab_size + i] - max_val);
        }
        
        // 2. Calcul du gradient dY = (Probas - Cible) / total_words
        for (int i = 0; i < vocab_size; i++) {
            float prob = expf(d_logits[idx * vocab_size + i] - max_val) / (sum_exp + 1e-7f);
            
            if (i == target_class) {
                d_dY[idx * vocab_size + i] = (prob - 1.0f) / total_words; // <-- AJOUT DE LA DIVISION
            } else {
                d_dY[idx * vocab_size + i] = (prob - 0.0f) / total_words; // <-- AJOUT DE LA DIVISION
            }
        }
    }
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 1. HYPERPARAMÈTRES INITIAUX
    int context_size = 32;
    int batch_size = 128;
    int embedding_dim = 128;
    int num_blocks = 4;
    float learning_rate = 3e-4f;
    int iterations = 10000;

    std::cout << "--- CHARGEMENT DES DONNEES ---" << std::endl;
    // Assure-toi d'avoir un fichier "input.txt" dans le même dossier !
    DataLoader dataloader("input.txt", batch_size, context_size);
    
    // Le DataLoader décide du vocab_size réel !
    int vocab_size = dataloader.get_vocab_size(); 

    std::cout << "--- CREATION DU MODELE GPT ---" << std::endl;
    GPTModel model(vocab_size, embedding_dim, batch_size, context_size, num_blocks);
    
    int total_words = batch_size * context_size;
    
    // 2. ALLOCATIONS MÉMOIRE
    // Mémoire CPU (Host)
    int* h_X = (int*)malloc(sizeof(int) * total_words);
    int* h_targets = (int*)malloc(sizeof(int) * total_words);

    // Mémoire GPU (Device)
    int* d_X;       
    int* d_targets; 
    float* d_dY;    
    cudaMalloc(&d_X, sizeof(int) * total_words);
    cudaMalloc(&d_targets, sizeof(int) * total_words);
    cudaMalloc(&d_dY, sizeof(float) * total_words * vocab_size);

    std::cout << "--- DEBUT DE L'ENTRAINEMENT ---" << std::endl;

    for (int iter = 0; iter < iterations; iter++) {
        
        dataloader.get_batch(h_X, h_targets);
        
        cudaMemcpy(d_X, h_X, sizeof(int) * total_words, cudaMemcpyHostToDevice);
        cudaMemcpy(d_targets, h_targets, sizeof(int) * total_words, cudaMemcpyHostToDevice);
        
        float* d_logits = model.forward(handle, d_X, 0);
        // Radar à NaN :
        int total_logits = total_words * vocab_size;
        check_nan_kernel<<<(total_logits + 255)/256, 256>>>(d_logits, total_logits, "SORTIE_LOGITS");
        cudaDeviceSynchronize();

        // NOUVEAU : Affichage de la Loss tous les 100 pas (Calcul sur CPU)
        if (iter % 100 == 0) {
            float* h_logits = (float*)malloc(sizeof(float) * total_words * vocab_size);
            cudaMemcpy(h_logits, d_logits, sizeof(float) * total_words * vocab_size, cudaMemcpyDeviceToHost);
            
            float loss = 0.0f;
            for(int i = 0; i < total_words; i++) {
                int target = h_targets[i];
                float max_l = -1e9f;
                for(int v=0; v<vocab_size; v++) max_l = std::max(max_l, h_logits[i*vocab_size + v]);
                
                float sum_exp = 0.0f;
                for(int v=0; v<vocab_size; v++) sum_exp += expf(h_logits[i*vocab_size + v] - max_l);
                
                float prob = expf(h_logits[i*vocab_size + target] - max_l) / sum_exp;
                loss += -logf(prob + 1e-7f); // Formule mathématique de la Cross-Entropy
            }
            loss /= total_words;
            std::cout << "Iteration " << iter << " / " << iterations << " | Loss: " << loss << std::endl;
            free(h_logits);
        }

        int threads = 256;
        int blocks = (total_words + threads - 1) / threads;
        cross_entropy_backward_kernel<<<blocks, threads>>>(d_logits, d_targets, d_dY, vocab_size, total_words);
        cudaDeviceSynchronize();

        // NOUVEAU : Le bouclier anti-explosion ! On limite l'erreur entre -1.0 et 1.0
        int total_grad_elements = total_words * vocab_size;
        int blocks_clip = (total_grad_elements + threads - 1) / threads;
        clip_gradients_kernel<<<blocks_clip, threads>>>(d_dY, -1.0f, 1.0f, total_grad_elements);
        cudaDeviceSynchronize();

        

        model.backward(handle, d_dY);
        model.step(learning_rate, iter + 1);
    }

    std::cout << "--- ENTRAINEMENT TERMINE ---" << std::endl;

    // --- TEST DE GÉNÉRATION ---
    // On extrait le dictionnaire pour la fonction generate_text
    std::map<int, char> int_to_char_map = dataloader.get_int_to_char_map();
    char* int_to_char_array = (char*)malloc(sizeof(char) * vocab_size);
    for(int i=0; i<vocab_size; i++) int_to_char_array[i] = int_to_char_map[i];

    std::cout << "\n--- GENERATION DE TEXTE ---" << std::endl;
    generate_text(handle, &model, "The city", 50, int_to_char_array, context_size, vocab_size);
    generate_text(handle, &model, "Romeo, ", 50, int_to_char_array, context_size, vocab_size);

    // NETTOYAGE
    free(h_X); free(h_targets); free(int_to_char_array);
    cudaFree(d_X); cudaFree(d_targets); cudaFree(d_dY);
    cublasDestroy(handle);

    return 0;
}
