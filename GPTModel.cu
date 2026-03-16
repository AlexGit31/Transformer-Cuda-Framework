#include "GPTModel.cuh"
#include <iostream>

// =========================================================================
// MÉTHODES DE LA CLASSE
// =========================================================================

GPTModel::GPTModel(int vocab_size, int embedding_dim, int batch_size, int context_size, int num_blocks) 
    : Layer(batch_size, context_size, embedding_dim) {
    
    this->vocab_size = vocab_size;
    this->embedding_dim = embedding_dim;
    this->batch_size = batch_size;
    this->context_size = context_size;
    this->num_blocks = num_blocks;

    // 1. Instanciation de l'Embedding
    embedding = new EmbeddingLayer(vocab_size, embedding_dim, batch_size, context_size);

    // 2. Instanciation de la tour de Blocs Transformer
    for (int i = 0; i < num_blocks; i++) {
        blocks.push_back(new TransformerBlock(batch_size, context_size, embedding_dim));
    }

    // 3. Instanciation de la RMSNorm finale
    final_norm = new RMSNormLayer(batch_size, context_size, embedding_dim);

    // 4. Instanciation de la LM Head
    // Entrée : Les 256 floats de la pensée finale du Transformer
    // Sortie : Les 65 floats de probabilité pour chaque lettre du vocabulaire
    int flat_batch = batch_size * context_size;
    lm_head = new LinearLayer(flat_batch, embedding_dim, vocab_size);
}

GPTModel::~GPTModel() {
    delete embedding;
    for (int i = 0; i < num_blocks; i++) {
        delete blocks[i];
    }
    delete final_norm;
    delete lm_head;
}

float* GPTModel::forward(cublasHandle_t handle, void* d_input, int activation_type) {
    // 1. On entre dans l'Embedding (d_input est notre int* du DataLoader)
    float* d_out = embedding->forward(handle, d_input, ACTIVATION_NONE);

    // 2. On traverse la tour de Blocs Transformer
    for (int i = 0; i < num_blocks; i++) {
        // La sortie du bloc i devient l'entrée du bloc i+1 !
        d_out = blocks[i]->forward(handle, d_out, ACTIVATION_NONE);
    }

    // 3. On stabilise une dernière fois
    d_out = final_norm->forward(handle, d_out, ACTIVATION_NONE);

    // 4. On projette vers le vocabulaire
    // d_out contient maintenant nos Logits ! (Dimension: batch_size * context_size * vocab_size)
    float* d_logits = lm_head->forward(handle, d_out, ACTIVATION_NONE);
    return d_logits;
}

float* GPTModel::backward(cublasHandle_t handle, float* d_dY) {
    // d_dY est l'erreur calculée par la fonction de perte (Cross-Entropy).
    
    // 1. Rétropropagation dans la tête de prédiction
    float* d_grad = lm_head->backward(handle, d_dY);

    // 2. Rétropropagation dans la norme finale
    d_grad = final_norm->backward(handle, d_grad);

    // 3. Rétropropagation dans la tour de Blocs (À L'ENVERS !)
    // On part du dernier bloc (num_blocks - 1) jusqu'au premier (0)
    for (int i = num_blocks - 1; i >= 0; i--) {
        d_grad = blocks[i]->backward(handle, d_grad);
    }

    // 4. Rétropropagation finale dans l'Embedding (qui met à jour le dictionnaire)
    embedding->backward(handle, d_grad);
    return nullptr; // Fin de la boucle pour le modèle
}

void GPTModel::step(float learning_rate, int t) {
    embedding->step(learning_rate, t);
    for (int i = 0; i < num_blocks; i++) {
        blocks[i]->step(learning_rate, t);
    }
    final_norm->step(learning_rate, t);
    lm_head->step(learning_rate, t);
}
