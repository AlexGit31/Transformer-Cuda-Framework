#include "AttentionLayer.cuh"
#include <iostream>
#include <cmath>

// =========================================================================
// KERNELS CUDA (Les petits outils de l'Attention)
// =========================================================================
// Kernel CUDA pour appliquer le Masque Causal (empêcher de tricher)
__global__ void causal_mask_kernel(float* d_Scores, int context_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = batch_size * context_size * context_size;
    
    if (idx < total_elements) {
        // On retrouve notre position dans la matrice 2D [context_size x context_size]
        int matrix_idx = idx % (context_size * context_size);
        int row = matrix_idx / context_size; // Le mot qui "regarde"
        int col = matrix_idx % context_size; // Le mot qui "est regardé"
        
        // Si on essaie de regarder dans le futur (colonne strictement supérieure à ligne)
        if (col > row) {
            d_Scores[idx] = -1e9f; // -Infini pour tuer l'attention vers le futur
        }
    }
}
// 1. Kernel pour diviser les scores par la racine carrée de la dimension
__global__ void scale_scores_kernel(float* d_Scores, float scale_factor, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        d_Scores[idx] = d_Scores[idx] * scale_factor;
    }
}
// Kernel CUDA pour un Softmax ultra-robuste (Forward)
// Kernel CUDA pour un Softmax ultra-robuste et SÉCURISÉ (Forward)
__global__ void softmax_forward_kernel(float* d_Scores, int context_size, int batch_size) {
    int row = blockIdx.x; 
    int tid = threadIdx.x; 
    
    // NOUVEAU : Mémoire partagée pour éviter que les threads se marchent dessus
    extern __shared__ float shared_exp[]; 
    
    if (row < batch_size * context_size && tid < context_size) {
        int base_idx = row * context_size;
        
        // 1. Recherche du Max (Rapide)
        float max_val = -1e9f;
        for (int i = 0; i < context_size; i++) {
            max_val = fmaxf(max_val, d_Scores[base_idx + i]);
        }
        
        // 2. Calcul de l'exponentielle stocké en lieu sûr !
        float my_exp = expf(d_Scores[base_idx + tid] - max_val);
        shared_exp[tid] = my_exp;
        
        // BARRIÈRE DE SÉCURITÉ : On attend que tout le monde ait posé son calcul
        __syncthreads(); 
        
        // 3. Calcul de la somme à partir de la mémoire partagée (protégée)
        float sum_exp = 0.0f;
        for (int i = 0; i < context_size; i++) {
            sum_exp += shared_exp[i];
        }
        
        // 4. Écriture finale (aucun risque de conflit)
        d_Scores[base_idx + tid] = my_exp / (sum_exp + 1e-9f);
    }
}
// 2. Kernel Softmax (Par ligne). 
// Chaque bloc gère UNE ligne de la matrice des scores (un mot qui regarde les autres)
__global__ void softmax_attention_kernel(float* d_Scores, int context_size) {
    int row_idx = blockIdx.x; // Quelle ligne (quel mot) on traite ?
    int col_idx = threadIdx.x; // Quelle colonne (quel mot on regarde) ?
    
    extern __shared__ float shared_scores[];

    if (col_idx < context_size) {
        int global_idx = row_idx * context_size + col_idx;
        float val = d_Scores[global_idx];
        
        // --- 1. Recherche du Max pour la stabilité numérique ---
        shared_scores[col_idx] = val;
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int index = 2 * stride * col_idx;
            if (index + stride < blockDim.x) {
                shared_scores[index] = fmaxf(shared_scores[index], shared_scores[index + stride]);
            }
            __syncthreads();
        }
        float max_val = shared_scores[0];
        __syncthreads();

        // --- 2. Exponentielle (e^x) ---
        val = expf(val - max_val);
        shared_scores[col_idx] = val;
        __syncthreads();

        // --- 3. Somme de la ligne ---
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int index = 2 * stride * col_idx;
            if (index + stride < blockDim.x) {
                shared_scores[index] += shared_scores[index + stride];
            }
            __syncthreads();
        }
        float sum = shared_scores[0];
        __syncthreads();

        // --- 4. Division finale (Pourcentage) ---
        d_Scores[global_idx] = val / (sum + 1e-9f);
    }
}


// Kernel CUDA pour la dérivée du Softmax (Backward)
__global__ void softmax_backward_kernel(float* d_dScores_softmax, float* d_Scores_softmax, float* d_dScores, int context_size) {
    int row_idx = blockIdx.x; 
    int col_idx = threadIdx.x;

    if (col_idx < context_size) {
        int global_idx = row_idx * context_size + col_idx;
        
        // 1. Calculer la somme (dScores_softmax * Scores_softmax) pour cette ligne
        float local_dot = d_dScores_softmax[global_idx] * d_Scores_softmax[global_idx];
        
        // (Pour faire simple sans mémoire partagée complète ici, 
        // on suppose une réduction classique ou une boucle sur la ligne)
        float row_sum = 0.0f;
        for(int i = 0; i < context_size; i++) {
            int i_idx = row_idx * context_size + i;
            row_sum += d_dScores_softmax[i_idx] * d_Scores_softmax[i_idx];
        }

        // 2. Appliquer la formule de la dérivée du Softmax
        d_dScores[global_idx] = d_Scores_softmax[global_idx] * (d_dScores_softmax[global_idx] - row_sum);
    }
}

// Kernel CUDA pour additionner les 3 gradients d'entrée
__global__ void sum_gradients_kernel(float* d_dX, const float* d_dX_q, const float* d_dX_k, const float* d_dX_v, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Sécurité classique
    if (idx < total_elements) {
        d_dX[idx] = d_dX_q[idx] + d_dX_k[idx] + d_dX_v[idx];
    }
}
// =========================================================================
// MÉTHODES DE LA CLASSE
// =========================================================================

AttentionLayer::AttentionLayer(int batch_size, int context_size, int embedding_dim) 
    : Layer(batch_size, context_size, embedding_dim) {
    
    this->batch_size = batch_size;
    this->context_size = context_size;
    this->embedding_dim = embedding_dim;

    // 1. Création de nos sous-couches LinearLayer.
    // L'astuce : on aplatit le batch_size et le context_size pour nos LinearLayers.
    int flat_batch = batch_size * context_size;
    
    W_q = new LinearLayer(flat_batch, embedding_dim, embedding_dim);
    W_k = new LinearLayer(flat_batch, embedding_dim, embedding_dim);
    W_v = new LinearLayer(flat_batch, embedding_dim, embedding_dim);
    W_o = new LinearLayer(flat_batch, embedding_dim, embedding_dim);

    // 2. Allocation des espaces mémoires de l'Attention
    // La matrice de Scores a pour taille [batch_size, context_size, context_size]
    cudaMalloc(&d_Scores, sizeof(float) * batch_size * context_size * context_size);
    
    // Le résultat (Scores * V) a pour taille [batch_size, context_size, embedding_dim]
    cudaMalloc(&d_AttentionOut, sizeof(float) * batch_size * context_size * embedding_dim);
    // NOUVEAU : Allocation VRAM pour la passe Backward
    int total_elements = batch_size * context_size * embedding_dim;
    int total_scores = batch_size * context_size * context_size;

    cudaMalloc(&d_Q_grad, sizeof(float) * total_elements);
    cudaMalloc(&d_K_grad, sizeof(float) * total_elements);
    cudaMalloc(&d_V_grad, sizeof(float) * total_elements);
    cudaMalloc(&d_dX, sizeof(float) * total_elements);

    cudaMalloc(&d_dScores_softmax, sizeof(float) * total_scores);
    cudaMalloc(&d_dScores, sizeof(float) * total_scores);
}

AttentionLayer::~AttentionLayer() {
    delete W_q;
    delete W_k;
    delete W_v;
    delete W_o;
    cudaFree(d_Scores);
    cudaFree(d_AttentionOut);
    // NOUVEAU : Nettoyage du Backward
    cudaFree(d_Q_grad);
    cudaFree(d_K_grad);
    cudaFree(d_V_grad);
    cudaFree(d_dX);
    cudaFree(d_dScores_softmax);
    cudaFree(d_dScores);
}

float* AttentionLayer::forward(cublasHandle_t handle, void* d_input, int activation_type) {
    float* d_X = (float*) d_input;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // =====================================================================
    // ÉTAPE 1 : Générer les matrices Q, K et V
    // =====================================================================
    // Nos LinearLayer gèrent déjà leurs propres allocations internes pour la sortie (d_Y).
    // On récupère juste les pointeurs !
    d_Q = W_q->forward(handle, d_X, ACTIVATION_NONE);
    d_K = W_k->forward(handle, d_X, ACTIVATION_NONE);
    d_V = W_v->forward(handle, d_X, ACTIVATION_NONE);

    // =====================================================================
    // ÉTAPE 2 : Produit Scalaire (Scores = Q * K^T)
    // =====================================================================
    // On utilise la fonction Batched pour ne pas mélanger les phrases du batch !
    // ATTENTION PIÈGE cuBLAS : cuBLAS lit en Column-Major. 
    // Pour calculer (Q * K^T) en Row-Major C++, on demande à cuBLAS de calculer (K^T * Q).
    
    long long int stride_Q = context_size * embedding_dim;
    long long int stride_K = context_size * embedding_dim;
    long long int stride_Scores = context_size * context_size;

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N, // Transposer K, Ne pas transposer Q
        context_size, context_size, embedding_dim, // m, n, k
        &alpha,
        d_K, embedding_dim, stride_K, // Matrice A (qui est K)
        d_Q, embedding_dim, stride_Q, // Matrice B (qui est Q)
        &beta,
        d_Scores, context_size, stride_Scores, // Matrice C (le résultat)
        batch_size // Le nombre de phrases indépendantes
    );

    // =====================================================================
    // ÉTAPE 3 : Mise à l'échelle (Scale)
    // =====================================================================
    int total_scores = batch_size * context_size * context_size;
    int threads_scale = 256;
    int blocks_scale = (total_scores + threads_scale - 1) / threads_scale;
    float scale_factor = 1.0f / sqrtf((float)embedding_dim); // 1.0 divisé par la racine !
    // Appel du kernel pour multiplier d_Scores par scale_factor
    
    scale_scores_kernel<<<blocks_scale, threads_scale>>>(d_Scores, scale_factor, total_scores);
    cudaDeviceSynchronize();

    // On applique le masque
    int threads_mask = 256;
    int blocks_mask = (total_scores + threads_mask - 1) / threads_mask;
    
    causal_mask_kernel<<<blocks_mask, threads_mask>>>(d_Scores, context_size, batch_size);
    cudaDeviceSynchronize();

    // =====================================================================
    // ÉTAPE 4 : Softmax (Les Pourcentages d'Attention)
    // =====================================================================
    int blocks_softmax = batch_size * context_size; // Une ligne = un bloc
    int threads_softmax = context_size; // Un thread par colonne (mot)
    size_t shared_mem = context_size * sizeof(float);
    
    softmax_attention_kernel<<<blocks_softmax, threads_softmax, shared_mem>>>(d_Scores, context_size,batch_size);
    cudaDeviceSynchronize();

    // =====================================================================
    // ÉTAPE 5 : Le Mélange (Output = Scores * V)
    // =====================================================================
    // Encore une inversion cuBLAS : (Scores * V) en Row-Major = (V * Scores) en Col-Major.
    long long int stride_V = context_size * embedding_dim;
    long long int stride_Out = context_size * embedding_dim;

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        embedding_dim, context_size, context_size, // m, n, k
        &alpha,
        d_V, embedding_dim, stride_V, // Matrice A (qui est V)
        d_Scores, context_size, stride_Scores, // Matrice B (qui est Scores)
        &beta,
        d_AttentionOut, embedding_dim, stride_Out, // Matrice C (Résultat)
        batch_size
    );

    // =====================================================================
    // ÉTAPE 6 : Projection finale (W_o)
    // =====================================================================
    return W_o->forward(handle, d_AttentionOut, ACTIVATION_NONE);
}

float* AttentionLayer::backward(cublasHandle_t handle, float* d_dY) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    long long int stride_Q = context_size * embedding_dim;
    long long int stride_Scores = context_size * context_size;

    // 1. Backward de W_o
    // (Suppose que tes LinearLayer::backward retournent float* d_dX)
    float* d_dOut = W_o->backward(handle, d_dY); 

    // 2. Backward de Output = Scores * V 
    // dV = Scores^T * dOut
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        embedding_dim, context_size, context_size, &alpha,
        d_dOut, embedding_dim, stride_Q,
        d_Scores, context_size, stride_Scores, &beta,
        d_V_grad, embedding_dim, stride_Q, batch_size); // Il faut un pointeur d_V_grad alloué !

    // dScores_softmax = dOut * V^T
    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        context_size, context_size, embedding_dim, &alpha,
        d_V, embedding_dim, stride_Q,
        d_dOut, embedding_dim, stride_Q, &beta,
        d_dScores_softmax, context_size, stride_Scores, batch_size);

    // 3. Backward du Softmax
    int blocks_softmax = batch_size * context_size;
    int threads_softmax = context_size;
    softmax_backward_kernel<<<blocks_softmax, threads_softmax>>>(d_dScores_softmax, d_Scores, d_dScores, context_size);
    cudaDeviceSynchronize();

    // 4. Backward du Scale (Mise à l'échelle)
    float scale_factor = 1.0f / sqrtf((float)embedding_dim);
    int total_scores = batch_size * context_size * context_size;
    int threads_scale = 256;
    int blocks_scale = (total_scores + threads_scale - 1) / threads_scale;
    // On réutilise scale_scores_kernel car c'est la même division mathématique !
    scale_scores_kernel<<<blocks_scale, threads_scale>>>(d_dScores, scale_factor, total_scores); 

    // 5. Backward de Scores = Q * K^T
    // dQ = dScores * K
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        embedding_dim, context_size, context_size, &alpha,
        d_K, embedding_dim, stride_Q,
        d_dScores, context_size, stride_Scores, &beta,
        d_Q_grad, embedding_dim, stride_Q, batch_size);

    // dK = dScores^T * Q
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        embedding_dim, context_size, context_size, &alpha,
        d_Q, embedding_dim, stride_Q,
        d_dScores, context_size, stride_Scores, &beta,
        d_K_grad, embedding_dim, stride_Q, batch_size);

    // 6. Backward des projections initiales W_q, W_k, W_v
    float* d_dX_q = W_q->backward(handle, d_Q_grad);
    float* d_dX_k = W_k->backward(handle, d_K_grad);
    float* d_dX_v = W_v->backward(handle, d_V_grad);

    // =====================================================================
    // ÉTAPE 7 : L'addition finale des gradients pour la couche précédente
    // =====================================================================
    int total_elements = batch_size * context_size * embedding_dim;
    int threads_sum = 256;
    int blocks_sum = (total_elements + threads_sum - 1) / threads_sum;

    // Lancement du kernel
    sum_gradients_kernel<<<blocks_sum, threads_sum>>>(
        d_dX,    // La destination (Le gradient final de la couche Attention)
        d_dX_q,  // Le gradient qui remonte de W_q
        d_dX_k,  // Le gradient qui remonte de W_k
        d_dX_v,  // Le gradient qui remonte de W_v
        total_elements
    );

    cudaDeviceSynchronize();
    return d_dX;
    }

void AttentionLayer::step(float learning_rate, int t) {
    W_q->step(learning_rate, t);
    W_k->step(learning_rate, t);
    W_v->step(learning_rate, t);
    W_o->step(learning_rate, t);
}
