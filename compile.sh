#!/bin/bash
# Compile GPT-CUDA v2
# Usage: bash compile.sh
# Requires: nvcc, CUDA toolkit, cuBLAS

echo "=== GPT-CUDA v2 Compilation ==="

nvcc -O3 -o gpt_cuda \
    main.cu \
    GPTModel.cu \
    TransformerBlock.cu \
    AttentionLayer.cu \
    FeedForwardLayer.cu \
    RMSNormLayer.cu \
    EmbeddingLayer.cu \
    LinearLayer.cu \
    DataLoader.cpp \
    -lcublas

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful → ./gpt_cuda"
    echo ""
    echo "To train:"
    echo "  1. Place your training text as 'input.txt'"
    echo "  2. Run: ./gpt_cuda"
    echo "  3. Training log saved to training_log.csv"
else
    echo "✗ Compilation failed"
    exit 1
fi
