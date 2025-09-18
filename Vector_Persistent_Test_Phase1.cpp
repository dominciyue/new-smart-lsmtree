#include "kvstore.h"
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
  int total = 128;

  std::vector<std::string> text = load_text("data/trimmed_text.txt");
  for (int i = 0; i < total; i++) {
    store.put(i, text[i]);
  }

  return 0;
}