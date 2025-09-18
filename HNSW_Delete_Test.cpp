#include "kvstore.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

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

  store.reset();

  bool pass = true;

  std::vector<std::string> text = load_text("data/trimmed_text.txt");

  std::unordered_map<std::string, std::uint64_t> content_to_id;

  int total = 128;

  for (int i = 0; i < total; i++) {
    store.put(i, text[i]);
    content_to_id[text[i]] = i;
  }

  int fake_random_delete_id = 114;
  std::string fake_random_delete_content = text[fake_random_delete_id];

  auto result = store.search_knn_hnsw(fake_random_delete_content, 3);

  if (result.size() != 3) {
    std::cout << "Error: result.size() != 3" << std::endl;
    pass = false;
  }

  for (auto &item : result) {
    store.del(item.first);
  }

  auto result2 = store.search_knn(fake_random_delete_content, 3);
  if (result2.size() != 3) {
    std::cout << "Error: result2.size() != 3" << std::endl;
    pass = false;
  }
  for (int i = 0; i < 3; i++) {
    if (result2[i].first == result[i].first) {
      std::cout << "Error: result2[" << i << "] is not deleted" << std::endl;
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
