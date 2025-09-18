#include "kvstore.h"
#include <iostream>
#include <fstream>
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
  store.reset();

  std::vector<std::string> text = load_text("data/trimmed_text.txt");
  int total = 128;
  for (int i = 0; i < total; i++) {
    store.put(i, text[i]);
  }

  // Delete the first half of the keys (0 to 63)
  for (int i = 0; i < total / 2; ++i) {
    store.del(i);
  }

  // Save the HNSW index (will contain nodes 64-127 as active)
  std::cout << "[INFO] Explicitly calling save_hnsw_index_to_disk..." << std::endl;
  store.save_hnsw_index_to_disk("hnsw_data"); // Removed trailing slash, as save_hnsw_index_to_disk handles path construction

  return 0;
}