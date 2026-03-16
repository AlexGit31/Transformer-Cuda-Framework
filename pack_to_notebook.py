import glob
import json
import os

# 1. On trouve tous les fichiers source de ton projet
extensions = ["*.cu", "*.cuh", "*.cpp", "*.h"]
files_to_pack = []
for ext in extensions:
    files_to_pack.extend(glob.glob(ext))

cells = []

# 2. On ajoute une cellule de titre sympa
cells.append(
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Mon GPT from scratch en C++ et CUDA\n",
            "Ce notebook génère automatiquement tous les fichiers source, télécharge les données, compile le projet et lance l'entraînement sur le GPU de Colab.",
        ],
    }
)

# 3. Pour chaque fichier, on crée une cellule avec %%writefile
for fname in files_to_pack:
    with open(fname, "r", encoding="utf-8") as f:
        # Jupyter aime avoir une liste de lignes se terminant par \n
        lines = f.readlines()

    source = [f"%%writefile {fname}\n"] + lines

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source,
        }
    )

# 4. LE BONUS : Une cellule pour télécharger Tiny Shakespeare automatiquement !
cells.append(
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Téléchargement du jeu de données Tiny Shakespeare\n",
            "!wget -O input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        ],
    }
)

# 5. La cellule finale pour la compilation et l'exécution !
compile_cmd = "!nvcc -O3 -o mon_gpt main.cu GPTModel.cu TransformerBlock.cu AttentionLayer.cu FeedForwardLayer.cu RMSNormLayer.cu EmbeddingLayer.cu LinearLayer.cu DataLoader.cpp -lcublas\n"
run_cmd = "!./mon_gpt\n"

cells.append(
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [compile_cmd, run_cmd],
    }
)

# 6. On assemble le dictionnaire JSON du Notebook
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Mon_GPT_CUDA.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

# 7. On sauvegarde le fichier .ipynb
with open("Mon_GPT_CUDA.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(
    f"✅ Succès ! Le notebook 'Mon_GPT_CUDA.ipynb' a été généré avec {len(files_to_pack)} fichiers C/CUDA."
)
