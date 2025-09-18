#include "kvstore.h"
#include <fstream>
#include <iostream>
#include <vector>

std::vector<std::string> load_text(std::string filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<std::string> text;
  while (std::getline(file, line)) {
    text.push_back(line);
  }
  return text;
}

bool check_result(std::vector<std::pair<std::uint64_t, std::string>> result,
                  std::string text) {
  for (int i = 0; i < result.size(); i++) {
    if (result[i].second == text) {
      return true;
    }
  }
  return false;
}

int main() {
  KVStore store("data/", "hnsw_data");

  // TODO: uncomment this line when you have implemented the function
  // store.load_hnsw_index_from_disk("hnsw_data/");

  int pass = 0;
  int total = 128;

  std::vector<std::string> text = load_text("data/trimmed_text.txt");
  for (int i = 0; i < total; i++) {
    std::vector<std::pair<std::uint64_t, std::string>> result =
        store.search_knn_hnsw(text[i], 3);
    if (result.size() != 3) {
      std::cout << "Error: result.size() != 3" << std::endl;
      continue;
    }
    if (!check_result(result, text[i])) {
      std::cout << "Error: value[" << i << "] is not correct" << std::endl;
      continue;
    }
    pass++;
  }

  double accept_rate = (double)pass / total;
  std::cout << "accept rate: " << accept_rate << std::endl;

  return 0;
}