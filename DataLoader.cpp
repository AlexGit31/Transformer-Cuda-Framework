#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>
#include <cstdlib>

DataLoader::DataLoader(const std::string& fp, int bs, int cs) {
    this->bs = bs; this->cs = cs;
    std::ifstream file(fp);
    if (!file.is_open()) { std::cerr << "ERROR: cannot open " << fp << std::endl; exit(1); }
    std::stringstream buf; buf << file.rdbuf(); raw_text = buf.str(); file.close();
    std::set<char> uniq(raw_text.begin(), raw_text.end());
    vs = uniq.size();
    int i = 0;
    for (char c : uniq) { c2i[c] = i; i2c[i] = c; i++; }
    tokens.reserve(raw_text.size());
    for (char c : raw_text) tokens.push_back(c2i[c]);
    std::cout << "DataLoader: " << tokens.size() << " chars, vocab=" << vs << std::endl;
}
DataLoader::~DataLoader() {}
void DataLoader::get_batch(int* X, int* Y) {
    for (int b = 0; b < bs; b++) {
        int start = rand() % (tokens.size() - cs - 1);
        for (int i = 0; i < cs; i++) {
            X[b * cs + i] = tokens[start + i];
            Y[b * cs + i] = tokens[start + i + 1];
        }
    }
}
