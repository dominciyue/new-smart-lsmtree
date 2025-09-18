#include "kvstore.h" // Assuming kvstore.h includes necessary HNSW types and KVStoreAPI
#include "utils.h"   // For any utility functions if needed, like utils::rmdir if cleaning up
#include <filesystem> // Added for std::filesystem

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip> // For std::fixed, std::setprecision

// Function to parse a line from embedding_100k.txt into a vector of floats
std::vector<float> parse_embedding_line(const std::string& line) {
    std::vector<float> vec;
    if (line.length() < 2 || line.front() != '[' || line.back() != ']') {
        std::cerr << "Warning: Malformed embedding line (missing brackets): " << line.substr(0, 50) << "..." << std::endl;
        return vec; // Return empty if malformed
    }

    std::string content = line.substr(1, line.length() - 2); // Remove brackets
    std::stringstream ss(content);
    std::string item;

    vec.reserve(768); // Pre-allocate for expected dimension

    while (std::getline(ss, item, ',')) {
        try {
            // Trim whitespace from item if any before converting
            size_t first_digit = item.find_first_not_of(" \\t\\n\\r\\f\\v");
            if (first_digit == std::string::npos) continue; // Skip if empty after trim
            size_t last_digit = item.find_last_not_of(" \\t\\n\\r\\f\\v");
            item = item.substr(first_digit, (last_digit - first_digit + 1));
            
            if (!item.empty()) {
                 vec.push_back(std::stof(item));
            }
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Warning: Invalid argument for std::stof: '" << item << "' in line content: " << content.substr(0, 50) << "..." << std::endl;
            // Optionally, clear vec and return, or skip this item
        } catch (const std::out_of_range& oor) {
            std::cerr << "Warning: Out of range for std::stof: '" << item << "' in line content: " << content.substr(0, 50) << "..." << std::endl;
        }
    }
    if (vec.size() != 768 && !vec.empty()) { // Check dimension if not empty
        std::cerr << "Warning: Parsed vector has dimension " << vec.size() << " but expected 768. Line: " << line.substr(0,70) << "..." << std::endl;
        // Depending on strictness, you might want to return an empty vector or throw an error
        // For now, we'll proceed with the parsed vector but this warning is important.
    }
    return vec;
}

int main(int argc, char* argv[]) {
    std::string data_dir = "./kvstore_data_large_test"; 
    std::string hnsw_index_save_path_serial = "./hnsw_data_large_serial";    // Path for serial save
    std::string hnsw_index_save_path_parallel = "./hnsw_data_large_parallel"; // Path for parallel save

    // Clean up previous run data for KVStore
    if (utils::dirExists(data_dir)) {
        try {
            std::filesystem::remove_all(data_dir);
            std::cout << "Cleaned up previous KVStore data directory: " << data_dir << std::endl;
        } catch(const std::filesystem::filesystem_error& err) {
            std::cerr << "Error removing directory " << data_dir << ": " << err.what() << std::endl;
        }
    }
    // Clean up previous HNSW index directories (serial and parallel)
    if (utils::dirExists(hnsw_index_save_path_serial)) {
         try {
            std::filesystem::remove_all(hnsw_index_save_path_serial);
            std::cout << "Cleaned up previous HNSW serial index directory: " << hnsw_index_save_path_serial << std::endl;
        } catch(const std::filesystem::filesystem_error& err) {
            std::cerr << "Error removing directory " << hnsw_index_save_path_serial << ": " << err.what() << std::endl;
        }
    }
    if (utils::dirExists(hnsw_index_save_path_parallel)) {
         try {
            std::filesystem::remove_all(hnsw_index_save_path_parallel);
            std::cout << "Cleaned up previous HNSW parallel index directory: " << hnsw_index_save_path_parallel << std::endl;
        } catch(const std::filesystem::filesystem_error& err) {
            std::cerr << "Error removing directory " << hnsw_index_save_path_parallel << ": " << err.what() << std::endl;
        }
    }
    utils::mkdir(data_dir.data());

    std::string text_file_path = "D:/lab-lsm-tree-handout/large_dataset/cleaned_text_100k.txt";
    std::string embedding_file_path = "D:/lab-lsm-tree-handout/large_dataset/embedding_100k.txt";

    std::ifstream text_file(text_file_path);
    std::ifstream embedding_file(embedding_file_path);

    if (!text_file.is_open()) {
        std::cerr << "Error: Could not open text file: " << text_file_path << std::endl;
        return 1;
    }
    if (!embedding_file.is_open()) {
        std::cerr << "Error: Could not open embedding file: " << embedding_file_path << std::endl;
        return 1;
    }

    std::cout << "Initializing KVStore..." << std::endl;
    // Pass hnsw_index_save_path as empty to avoid loading an index during this phase of populating.
    // Or, if your KVStore constructor doesn't take HNSW path, that's fine.
    // The crucial part is that save_hnsw_index_to_disk will use the specified path.
    KVStore kvstore(data_dir, ""); // Provide KVStore data directory, no initial HNSW load path

    std::string sentence;
    std::string embedding_line;
    uint64_t key = 0;
    const int report_interval = 1000; // Report progress every 1000 items

    std::cout << "Starting to load 100k items..." << std::endl;
    auto load_start_time = std::chrono::high_resolution_clock::now();

    while (std::getline(text_file, sentence) && std::getline(embedding_file, embedding_line)) {
        if (sentence.empty() && embedding_line.empty()) continue; // Skip blank lines if any

        std::vector<float> vec = parse_embedding_line(embedding_line);

        if (vec.empty() || vec.size() != 768) { // Ensure vector is valid and has correct dimension
            std::cerr << "Skipping item for key " << key << " due to parsing error or incorrect dimension (" << vec.size() << ")." << std::endl;
            if(vec.empty() && !embedding_line.empty()) { // If parsing returned empty but line was not empty
                 std::cerr << "   Original embedding line (first 70 chars): " << embedding_line.substr(0, 70) << "..." << std::endl;
            }
            key++; // Increment key anyway to maintain sync if one file has an extra line (though ideally they are perfectly synced)
            continue;
        }
        
        // The put_with_precomputed_embedding function in your kvstore.cc
        // should handle setting the embedding_dimension_ if it's the first time.
        kvstore.put_with_precomputed_embedding(key, sentence, vec);

        if ((key + 1) % report_interval == 0) {
            std::cout << "Loaded " << (key + 1) << " items..." << std::endl;
        }
        key++;
    }

    auto load_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_duration = load_end_time - load_start_time;
    std::cout << "Finished loading " << key << " items in " 
              << std::fixed << std::setprecision(2) << load_duration.count() << " seconds." << std::endl;

    text_file.close();
    embedding_file.close();

    if (key == 0) {
        std::cerr << "No items were loaded. Please check the dataset files and paths." << std::endl;
        return 1;
    }

    // --- SERIAL HNSW SAVE TEST ---
    std::cout << "\n--- Starting HNSW Index SERIAL Save Test ---" << std::endl;
    auto serial_save_start_time = std::chrono::high_resolution_clock::now();
    kvstore.save_hnsw_index_to_disk(hnsw_index_save_path_serial, true); // force_serial = true
    auto serial_save_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serial_save_duration = serial_save_end_time - serial_save_start_time;

    // --- PARALLEL HNSW SAVE TEST ---
    // It's good practice to re-create or ensure the directory for parallel save is clean too,
    // although the save function itself creates directories. Let's be explicit.
    if (utils::dirExists(hnsw_index_save_path_parallel)) { // Should have been cleaned at start, but good for clarity
         try {
            std::filesystem::remove_all(hnsw_index_save_path_parallel);
            // std::cout << "Re-cleaned HNSW parallel index directory: " << hnsw_index_save_path_parallel << std::endl;
        } catch(const std::filesystem::filesystem_error& err) {
            std::cerr << "Error re-removing parallel directory " << hnsw_index_save_path_parallel << ": " << err.what() << std::endl;
        }
    }
    std::cout << "\n--- Starting HNSW Index PARALLEL Save Test ---" << std::endl;
    auto parallel_save_start_time = std::chrono::high_resolution_clock::now();
    kvstore.save_hnsw_index_to_disk(hnsw_index_save_path_parallel, false); // force_serial = false (or default)
    auto parallel_save_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parallel_save_duration = parallel_save_end_time - parallel_save_start_time;


    std::cout << "\n-----------------------------------------------------" << std::endl;
    std::cout << "HNSW Index SERIAL Save Time (for " << key << " items): "
              << std::fixed << std::setprecision(4) << serial_save_duration.count() << " seconds." << std::endl;
    std::cout << "Serial Index saved to: " << hnsw_index_save_path_serial << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "HNSW Index PARALLEL Save Time (for " << key << " items): "
              << std::fixed << std::setprecision(4) << parallel_save_duration.count() << " seconds." << std::endl;
    std::cout << "Parallel Index saved to: " << hnsw_index_save_path_parallel << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "KVStore data (SSTables, embeddings.bin) in: " << data_dir << std::endl;

    std::cout << "\nTest finished." << std::endl;

    return 0;
} 