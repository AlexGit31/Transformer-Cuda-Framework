#include "FeedForwardLayer.cuh"
#include <iostream>

// =========================================================================
// MÉTHODES DE LA CLASSE
// =========================================================================

FeedForwardLayer::FeedForwardLayer(int batch_size, int context_size, int embedding_dim, int expansion_factor) 
    : Layer(batch_size, context_size, embedding_dim) {
    
    this->batch_size = batch_size;
    this->context_size = context_size;
    this->embedding_dim = embedding_dim;
    
    // Le secret du Transformer : La couche cachée est 4 fois plus grande !
    this->hidden_dim = embedding_dim * expansion_factor;

    // Astuce classique : on aplatit le temps et le batch
    int flat_batch = batch_size * context_size;

    // Instanciation de nos deux couches
    // Couche 1 : Entrée = embedding_dim, Sortie = hidden_dim (ex: 256 -> 1024)
    fc1 = new LinearLayer(flat_batch, embedding_dim, hidden_dim);
    
    // Couche 2 : Entrée = hidden_dim, Sortie = embedding_dim (ex: 1024 -> 256)
    fc2 = new LinearLayer(flat_batch, hidden_dim, embedding_dim);
}

FeedForwardLayer::~FeedForwardLayer() {
    delete fc1;
    delete fc2;
}

float* FeedForwardLayer::forward(cublasHandle_t handle, void* d_input, int activation_type) {
    // Étape 1 : Expansion avec activation (Le ReLU coupe les valeurs négatives)
    // Note : On utilise l'activation DANS la première couche
    float* d_hidden = fc1->forward(handle, (float*)d_input, ACTIVATION_RELU);

    // Étape 2 : Contraction vers la taille d'origine (Pas d'activation ici !)
    float* d_output = fc2->forward(handle, d_hidden, ACTIVATION_NONE);

    return d_output;
}

float* FeedForwardLayer::backward(cublasHandle_t handle, float* d_dY) {
    // La rétropropagation est magique grâce à notre architecture objet :
    // 1. Le gradient traverse la couche 2 (qui nous renvoie le gradient intermédiaire)
    float* d_dHidden = fc2->backward(handle, d_dY);

    // 2. Ce gradient intermédiaire traverse la couche 1 (qui nous renvoie le dX final)
    float* d_dX = fc1->backward(handle, d_dHidden);
    
    // (Dans un code ultra-complet, on stockerait d_dX dans la classe pour le retourner
    // à la couche d'Attention qui se trouve en dessous)
    return d_dX;
}

void FeedForwardLayer::step(float learning_rate, int t) {
    fc1->step(learning_rate, t);
    fc2->step(learning_rate, t);
}
