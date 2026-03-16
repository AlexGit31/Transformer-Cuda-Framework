#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>
#include <cstdlib>

DataLoader::DataLoader(const std::string& filepath, int batch_size, int context_size) {
    this->batch_size = batch_size;
    this->context_size = context_size;

    // 1. Lecture du fichier texte
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERREUR CRITIQUE : Impossible d'ouvrir le fichier " << filepath << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    raw_text = buffer.str();
    file.close();

    // 2. Création du vocabulaire (Trouver les caractères uniques)
    std::set<char> unique_chars(raw_text.begin(), raw_text.end());
    vocab_size = unique_chars.size();

    // 3. Remplissage des dictionnaires
    int i = 0;
    for (char c : unique_chars) {
        char_to_int[c] = i;
        int_to_char[i] = c;
        i++;
    }

    // 4. Tokenization : On convertit tout le texte en chiffres !
    tokens.reserve(raw_text.size());
    for (char c : raw_text) {
        tokens.push_back(char_to_int[c]);
    }

    std::cout << "DataLoader initialise ! Taille du texte: " << tokens.size() 
              << " caracteres, Vocabulaire: " << vocab_size << " caracteres." << std::endl;
}

DataLoader::~DataLoader() {}

void DataLoader::get_batch(int* h_X, int* h_targets) {
    // Pour chaque ligne du batch
    for (int b = 0; b < batch_size; b++) {
        // On tire un index de départ au hasard (en laissant la place pour le context_size + la cible)
        int max_start_idx = tokens.size() - context_size - 1;
        int start_idx = rand() % max_start_idx;

        // On remplit le contexte (X) et la cible décalée d'un cran (Y)
        for (int i = 0; i < context_size; i++) {
            h_X[b * context_size + i] = tokens[start_idx + i];
            h_targets[b * context_size + i] = tokens[start_idx + i + 1];
        }
    }
}
