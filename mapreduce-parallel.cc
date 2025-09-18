#include <algorithm>
#include <future> // For std::async and std::future
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

std::string clean_word(std::string word) {
  std::transform(word.begin(), word.end(), word.begin(), ::tolower);
  word.erase(std::remove_if(word.begin(), word.end(),
                            [](char c) { return !std::isalpha(c); }),
             word.end());
  return word;
}

using KeyValue = std::pair<std::string, int>;
using IntermediateValues = std::vector<int>;

class WordCountMapper {
public:
  std::vector<KeyValue> map(const std::string &document_line) {
    std::vector<KeyValue> mapped_pairs;
    std::stringstream ss(document_line);
    std::string word;

    std::cout << "Mapper working on thread: " << std::this_thread::get_id()
              << " for line: " << document_line.substr(0, 10) << "..."
              << std::endl;

    while (ss >> word) {
      std::string cleaned = clean_word(word);
      if (!cleaned.empty()) {
        mapped_pairs.push_back({cleaned, 1});
      }
    }
    return mapped_pairs;
  }
};

class WordCountReducer {
public:
  KeyValue reduce(const std::string &key, const IntermediateValues &values) {
    int sum = 0;
    std::cout << "Reducer working on thread: " << std::this_thread::get_id()
              << " for " << values.size() << " values" << std::endl;
    for (int val : values) {
      sum += val;
    }
    return {key, sum};
  }
};

int main() {
  // 0. 输入数据
  std::vector<std::string> input_documents = {
      "Hello World this is a test",
      "Hello again World this is another test!",
      "Test test test and hello one more time.",
      "C++ is fun and C++ can be fast",
      "MapReduce with C++ and threads",
      "Another line for more data processing",
      "Parallel processing is the key for speed",
      "Speed and efficiency are important"};

  WordCountMapper mapper_instance;
  WordCountReducer reducer_instance;

  // --- 1. 并行 Map 阶段 ---
  std::cout << "--- Parallel Map Phase ---" << std::endl;
  std::vector<std::future<std::vector<KeyValue>>> map_futures;

  for (const std::string &line : input_documents) {
    map_futures.push_back(
        std::async(std::launch::async, [&mapper_instance, line]() {
          return mapper_instance.map(line);
        }));
  }

  std::vector<KeyValue> mapped_output;
  for (auto &fut : map_futures) {
    std::vector<KeyValue> pairs_from_task = fut.get();
    mapped_output.insert(mapped_output.end(), pairs_from_task.begin(),
                         pairs_from_task.end());
  }
  std::cout << "Map phase completed. Total mapped pairs: "
            << mapped_output.size() << std::endl;
  std::cout << std::endl;

  // --- 2. Shuffle and Sort 阶段 ---
  std::cout << "--- Shuffle/Group Phase ---" << std::endl;
  std::map<std::string, IntermediateValues> grouped_values;
  for (const auto &pair : mapped_output) {
    grouped_values[pair.first].push_back(pair.second);
  }

  std::cout << "Grouping phase completed. Total unique keys: "
            << grouped_values.size() << std::endl;
  std::cout << std::endl;

  // --- 3. 并行 Reduce 阶段 ---
  std::cout << "--- Parallel Reduce Phase ---" << std::endl;
  std::vector<std::future<KeyValue>> reduce_futures;

  for (const auto &group : grouped_values) {
    reduce_futures.push_back(
        std::async(std::launch::async, [&reducer_instance, key = group.first,
                                        values = group.second]() {
          return reducer_instance.reduce(key, values);
        }));
  }

  std::vector<KeyValue> reduced_output;
  for (auto &fut : reduce_futures) {
    KeyValue result = fut.get(); // 等待任务完成并获取结果
    reduced_output.push_back(result);
  }
  std::cout << "Reduce phase completed." << std::endl;
  std::cout << std::endl;

  // --- 4. 最终结果 ---
  std::cout << "--- Final Word Counts ---" << std::endl;
  std::sort(
      reduced_output.begin(), reduced_output.end(),
      [](const KeyValue &a, const KeyValue &b) { return a.first < b.first; });

  for (const auto &pair : reduced_output) {
    std::cout << pair.first << ": " << pair.second << std::endl;
  }

  return 0;
}