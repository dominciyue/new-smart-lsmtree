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

int main() {
  KVStore store("data/");

  // TODO: uncomment this line when you have implemented the function
  store.load_embedding_from_disk("data/");

  bool pass = true;

  std::vector<std::string> text = load_text("data/trimmed_text.txt");
  int total = 128;

  for (int i = 0; i < total; i++) {
    std::vector<std::pair<std::uint64_t, std::string>> result =
        store.search_knn(text[i], 1);
    if (result.size() != 1) {
      std::cout << "Error: result.size() != 1" << std::endl;
      pass = false;
      continue;
    }
    if (result[0].second != text[i]) {
      std::cout << "Error: value[" << i << "] is not correct" << std::endl;
      pass = false;
    }
  }

  if (pass) {
    std::cout << "Test passed" << std::endl;
  } else {
    std::cout << "Test failed" << std::endl;
  }

  return 0;
}