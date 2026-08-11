#pragma once
#include <string>
#include <vector>
#include <map>
class DataLoader {
  std::string raw_text; std::vector<int> tokens;
  std::map<char,int> c2i; std::map<int,char> i2c;
  int bs, cs, vs;
public:
  DataLoader(const std::string& fp, int bs, int cs);
  ~DataLoader();
  void get_batch(int* X, int* Y);
  int get_vocab_size() const { return vs; }
  int get_num_tokens() const { return (int)tokens.size(); }
  std::map<int,char> get_i2c() const { return i2c; }
};
