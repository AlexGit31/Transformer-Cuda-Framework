# 🧠 GPT-CUDA: A "From Scratch" Transformer in C++ and CUDA

![C++](https://img.shields.io/badge/C++-17-blue.svg) ![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg) ![cuBLAS](https://img.shields.io/badge/Library-cuBLAS-orange.svg) ![Status](https://img.shields.io/badge/Status-Functional-brightgreen.svg)

**GPT-CUDA** is an educational and highly optimized implementation of a Generative Pre-trained Transformer (GPT) language model, written **entirely from scratch in C++ and CUDA**, without relying on any Deep Learning frameworks (no PyTorch, no TensorFlow).

This project was built to break the "black box" of LLMs and understand the intimate, hardware-level mechanics of neural networks: VRAM management, memory coalescing, thread synchronization, and the raw mathematics of backpropagation.

## 🚀 Features

- **Full Transformer Architecture:** Faithful implementation of the original _Attention Is All You Need_ paper.
- **Custom CUDA Kernels:** All compute-heavy operations (Softmax, Causal Masking, RMSNorm, AdamW) are written as custom CUDA kernels leveraging `__shared__` memory for maximum performance.
- **Integrated AdamW Optimizer:** An adaptive optimizer featuring Momentum, Velocity, and Weight Decay, coded entirely from scratch.
- **Extreme Numerical Stability:** Built-in protections against exploding gradients and numerical overflows (Gradient Clipping, Max-Trick for Softmax, and rigorous `NaN` handling).
- **Autoregressive Generation:** Capable of generating text character-by-character (or token-by-token) using probabilistic sampling.

---

## 🏗️ Architecture and Components (Deep Dive)

The project relies on an Object-Oriented architecture in C++. Each layer inherits from an abstract `Layer` base class defining a strict contract: `forward`, `backward`, and `step`.

### 1. Embedding and Positional Encoding (`EmbeddingLayer.cu`)

- **Word Dictionary:** A matrix projecting token IDs into a dense vector space (`embedding_dim`).
- **Sinusoidal Positional Encoding:** Instead of learning positions, the model uses mathematical Fourier waves (Sine/Cosine) pre-computed on the CPU and transferred to the GPU. These waves are fused with the token embeddings via an ultra-fast CUDA kernel.

### 2. The Core Engine: Attention (`AttentionLayer.cu`)

The Attention mechanism calculates the relationships between every word in the sequence.

- **cuBLAS Sgemm:** Massive matrix multiplications ($Q \times K^T$ and $Scores \times V$) are delegated to NVIDIA's hyper-optimized library (via `cublasSgemmStridedBatched`).
- **Causal Masking:** A dedicated CUDA kernel fills the upper triangle of the attention matrix with `-1e9f` to prevent the model from "cheating" by looking into the future during training.
- **Fused Safe-Softmax:** Softmax is a major point of failure in CUDA. Our kernel uses `__shared__` memory to perform a parallel reduction, find the maximum value (Max-Trick), and calculate the exponential without ever risking an `Overflow / NaN`.

### 3. Normalization and FeedForward (`RMSNormLayer.cu` & `FeedForwardLayer.cu`)

- **RMSNorm:** Implementation of Root Mean Square Normalization. The Backward pass uses the exact, complex mathematical derivative (including the negative restoring force term) to prevent Catastrophic Collapse of the activations.
- **MLP (FeedForward):** A classic neural network expanding the hidden dimension by a factor of 4, followed by a ReLU activation.

### 4. The Survival Shield: Residual Connections (`TransformerBlock.cu`)

The network integrates Skip Connections (`Output = Input + Layer(Input)`) via custom `add_tensors_kernel` addition kernels. This entirely prevents the Vanishing Gradient problem across the deep layers of the Transformer tower.

---

## 💥 War Stories: What I Learned

Building an LLM from scratch means hitting mathematical brick walls. Here are the industrial-grade bugs resolved in this repository:

1. **Adam's Coma (Loss stuck at 4.17):** Without positional encoding or causal masking, the network is just a "bag of words" incapable of learning, locking the Loss at the absolute random chance score of $-\ln(1/65)$.
2. **Variance Explosion:** Forgetting the `1.0f / sqrt(dim)` scaling factor during the Attention backward pass multiplies gradients by 128, creating infinite Velocities in the AdamW optimizer.
3. **Catastrophic Collapse:** Using a "Straight-Through Estimator" (ignoring the derivative) on RMSNorm causes activations to grow until they exceed the `Float32` limit ($10^{38}$), generating unrecoverable `NaN`s.
4. **Softmax Race Condition:** Failing to synchronize CUDA threads (`__syncthreads()`) when calculating the sum of exponentials corrupts the probability distribution.

---

## 🛠️ Build and Run

### Prerequisites

- A CUDA-capable NVIDIA GPU.
- The CUDA Toolkit installed (`nvcc`).
- A raw text file named `input.txt` (e.g., Tiny Shakespeare) in the root directory.

### Compilation

Compile the project and link the cuBLAS library:

```bash
nvcc -o gpt main.cu Layer.cu LinearLayer.cu EmbeddingLayer.cu AttentionLayer.cu RMSNormLayer.cu FeedForwardLayer.cu TransformerBlock.cu GPTModel.cu -lcublas -O3
```

### Execution

```bash
./gpt
```

The model will train for 10,000 iterations (configurable in main.cu) and automatically generate text upon completion.

## 🤝 Contributing & Acknowledgments

This project is an educational demonstration of the raw power of C++ and CUDA for Artificial Intelligence. Feel free to fork, explore the kernels, and submit PRs to optimize VRAM performance!

Inspired by the original GPT architecture and the "Attention Is All You Need" paper (Vaswani et al., 2017).
