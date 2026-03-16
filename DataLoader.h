#pragma once
#include <string>
#include <vector>
#include <map>

class DataLoader {
private:
    std::string raw_text;
    std::vector<int> tokens; // Le texte entier converti en nombres
    
    // Nos dictionnaires de traduction
    std::map<char, int> char_to_int;
    std::map<int, char> int_to_char;
    
    int batch_size;
    int context_size;
    int vocab_size;

public:
    DataLoader(const std::string& filepath, int batch_size, int context_size);
    ~DataLoader();

    // Remplit les tableaux CPU avec un nouveau batch tiré au hasard
    void get_batch(int* h_X, int* h_targets);
    
    // Getters utiles pour configurer le modèle
    int get_vocab_size() const { return vocab_size; }
    std::map<int, char> get_int_to_char_map() const { return int_to_char; }
};
