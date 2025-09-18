#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

void create_dummy_file(const std::string &filename, size_t num_long_ints) {
  std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);
  if (!outfile) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }
  for (size_t i = 0; i < num_long_ints; ++i) {
    long int val = static_cast<long int>(i);
    outfile.write(reinterpret_cast<const char *>(&val), sizeof(long int));
  }
  outfile.close();
  std::cout << "Created dummy file '" << filename << "' with " << num_long_ints
            << " long ints." << std::endl;
}

std::vector<long int> read_chunk(const std::string &filename,
                                 long long offset_bytes,
                                 long long bytes_to_read) {
  std::vector<long int> local_data;
  if (bytes_to_read == 0) {
    return local_data;
  }

  std::ifstream infile(filename, std::ios::binary);
  if (!infile) {
    std::cerr << "Thread: Cannot open file: " << filename << std::endl;
    return local_data;
  }

  infile.seekg(offset_bytes, std::ios::beg);
  if (!infile) {
    std::cerr << "Thread: Cannot seek to offset " << offset_bytes << std::endl;
    return local_data;
  }

  size_t num_long_ints_in_chunk = bytes_to_read / sizeof(long int);
  local_data.resize(num_long_ints_in_chunk);

  infile.read(reinterpret_cast<char *>(local_data.data()), bytes_to_read);

  if (static_cast<long long>(infile.gcount()) != bytes_to_read) {
    std::cerr << "Thread: Read error. Expected " << bytes_to_read
              << " bytes, but got " << infile.gcount() << " bytes."
              << std::endl;
    local_data.resize(infile.gcount() / sizeof(long int));
  }
  infile.close();
  return local_data;
}

int main() {
  const std::string filename = "large_data.bin";
  const size_t num_long_ints_total = 10000000; // 10 million long ints

  // 1. Create a dummy file for demonstration
  try {
    create_dummy_file(filename, num_long_ints_total);
  } catch (const std::runtime_error &e) {
    std::cerr << "Error creating file: " << e.what() << std::endl;
    return 1;
  }

  // 2. Get file size
  long long file_size_bytes;
  try {
    file_size_bytes = std::filesystem::file_size(filename);
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "Error getting file size: " << e.what() << std::endl;
    return 1;
  }

  if (file_size_bytes == 0) {
    std::cout << "File is empty." << std::endl;
    return 0;
  }
  if (file_size_bytes % sizeof(long int) != 0) {
    std::cerr << "Warning: File size is not a multiple of sizeof(long int)."
              << std::endl;
    // Decide how to handle this, e.g., truncate or error
  }

  size_t actual_num_long_ints = file_size_bytes / sizeof(long int);
  std::cout << "File size: " << file_size_bytes << " bytes ("
            << actual_num_long_ints << " long ints)." << std::endl;
  std::cout << "Size of long int: " << sizeof(long int) << " bytes."
            << std::endl;

  // 3. Determine number of threads and chunk size
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 2; // Default if hardware_concurrency is not informative
  std::cout << "Using " << num_threads << " threads." << std::endl;

  long long bytes_per_thread_ideal = file_size_bytes / num_threads;
  // Align to sizeof(long int)
  long long bytes_per_thread_aligned =
      (bytes_per_thread_ideal / sizeof(long int)) * sizeof(long int);
  if (bytes_per_thread_aligned == 0 &&
      file_size_bytes >
          0) { // Ensure at least one long int per thread if possible
    bytes_per_thread_aligned = sizeof(long int);
  }

  std::vector<std::future<std::vector<long int>>> futures;
  long long current_offset_bytes = 0;

  for (unsigned int i = 0; i < num_threads; ++i) {
    long long bytes_for_this_thread = bytes_per_thread_aligned;

    // For the last thread, assign all remaining bytes
    if (i == num_threads - 1) {
      bytes_for_this_thread = file_size_bytes - current_offset_bytes;
    }

    // Ensure we don't try to read past the end or read 0 bytes if not necessary
    if (current_offset_bytes >= file_size_bytes) {
      break; // No more data to read
    }
    if (bytes_for_this_thread <= 0 &&
        file_size_bytes > 0) { // if remaining bytes is less than sizeof(long
                               // int) for last thread
      if (file_size_bytes - current_offset_bytes > 0) {
        bytes_for_this_thread = file_size_bytes - current_offset_bytes;
        // This might be smaller than sizeof(long int), read_chunk needs to
        // handle it or we error out
        if (bytes_for_this_thread % sizeof(long int) != 0) {
          std::cerr << "Error: Last chunk has misaligned size. Skipping."
                    << std::endl;
          bytes_for_this_thread =
              (bytes_for_this_thread / sizeof(long int)) * sizeof(long int);
        }
      } else {
        break;
      }
    }

    if (bytes_for_this_thread > 0) {
      std::cout << "Thread " << i << " reading from " << current_offset_bytes
                << " for " << bytes_for_this_thread << " bytes." << std::endl;
      futures.push_back(std::async(std::launch::async, read_chunk, filename,
                                   current_offset_bytes,
                                   bytes_for_this_thread));
      current_offset_bytes += bytes_for_this_thread;
    }
  }

  // 4. Collect results
  std::vector<long int> all_data;
  // Pre-allocate space if you know the total number of long ints
  all_data.reserve(actual_num_long_ints);

  for (size_t i = 0; i < futures.size(); ++i) {
    try {
      std::vector<long int> thread_data = futures[i].get();
      all_data.insert(all_data.end(), thread_data.begin(), thread_data.end());
    } catch (const std::exception &e) {
      std::cerr << "Exception caught from thread: " << e.what() << std::endl;
      // Handle error, maybe exit or try to continue
    }
  }

  std::cout << "Successfully read " << all_data.size() << " long ints in total."
            << std::endl;

  // 5. Verification (optional)
  if (all_data.size() == actual_num_long_ints) {
    bool ok = true;
    for (size_t i = 0; i < all_data.size(); ++i) {
      if (all_data[i] != static_cast<long int>(i)) {
        std::cout << "Verification failed at index " << i << ": expected " << i
                  << ", got " << all_data[i] << std::endl;
        ok = false;
        break;
      }
    }
    if (ok) {
      std::cout << "Verification successful." << std::endl;
    }
  } else {
    std::cout << "Verification skipped: Mismatch in number of elements read vs "
                 "expected."
              << std::endl;
  }

  // Clean up dummy file
  // std::filesystem::remove(filename);

  return 0;
}