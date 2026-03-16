#pragma once // <-- INDISPENSABLE pour éviter la redéfinition !
#include <cublas_v2.h>

// Définitions de nos constantes d'activation
#define ACTIVATION_NONE 0
#define ACTIVATION_RELU 1
#define ACTIVATION_GELU 2

class Layer {
  protected:
    int batch_features;
    int in_features;
    int out_features;

  public:
    // Constructeur de base
    Layer(int batch_size, int in_feat, int out_feat) {
        this->batch_features = batch_size;
        this->in_features = in_feat;
        this->out_features = out_feat;
    }

    // Destructeur virtuel obligatoire quand on fait de l'héritage
    virtual ~Layer() {}

    // LE CONTRAT (Pure virtual functions)
    // ATTENTION : On utilise bien un void* pour d_input ici !
    virtual float* forward(cublasHandle_t handle, void* d_input, int activation_type) = 0; 
    virtual float* backward(cublasHandle_t handle, float* d_dY) = 0;
    virtual void step(float learning_rate,int t) = 0;
};
