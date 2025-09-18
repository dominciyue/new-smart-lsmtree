#include "../test.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <set>

#include <cassert>
#include <cstdint>


std::vector<std::string> read_file(std::string filename) {
	std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr<<"Failed to open file: "<<filename<<std::endl;
        return {};
    }
    std::string line;
    std::vector<std::string> temp;
    while (std::getline(file, line)) {
        bool exist_alpha = false;
        for (auto c : line) {
            if (isalpha(c)) {
                exist_alpha = true;
                break;
            }
        }
        if (!exist_alpha) {
            continue;
        }
        if (line.empty())
            continue;
        if(line.size() < 70) {
            continue;
        }
        temp.push_back(line);
    }
    file.close();
    return temp;
}

class CorrectnessTest : public Test {
private:
    const uint64_t SIMPLE_TEST_MAX = 512;
    const uint64_t MIDDLE_TEST_MAX  = 1024 * 64;
    const uint64_t LARGE_TEST_MAX  = 1024 * 64;

    long long put_duration_ns = 0;
    long long get_duration_ns = 0;
    long long total_embedding_ns = 0;
    long long total_hnsw_search_only_ns = 0;
    long long total_knn_search_only_ns = 0;

    uint64_t put_call_count = 0;
    uint64_t get_call_count = 0;
    uint64_t get_embedding_call_count = 0;
    uint64_t search_hnsw_vec_call_count = 0;
    uint64_t search_knn_vec_call_count = 0;

    uint64_t total_ground_truth_results = 0;
    uint64_t total_matches_found = 0;

	void text_test(uint64_t max) {
		uint64_t i;
		auto trimmed_text = read_file("./data/trimmed_text.txt");
		max = std::min(max, (uint64_t)trimmed_text.size());

        put_call_count = 0;
        get_call_count = 0;
        get_embedding_call_count = 0;
        search_hnsw_vec_call_count = 0;
        search_knn_vec_call_count = 0;
        total_ground_truth_results = 0;
        total_matches_found = 0;
        put_duration_ns = 0;
        get_duration_ns = 0;
        total_embedding_ns = 0;
        total_hnsw_search_only_ns = 0;
        total_knn_search_only_ns = 0;

        auto put_start = std::chrono::high_resolution_clock::now();
		for (i = 0; i < max; ++i) {
			store.put(i, trimmed_text[i]);
            put_call_count++;
		}
        auto put_end = std::chrono::high_resolution_clock::now();
        put_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(put_end - put_start).count();

        auto get_start = std::chrono::high_resolution_clock::now();
		for (i = 0; i < max; ++i) {
			EXPECT(trimmed_text[i], store.get(i));
            get_call_count++;
        }
        auto get_end = std::chrono::high_resolution_clock::now();
        get_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(get_end - get_start).count();

		auto test_text = read_file("./data/test_text.txt");
		max = std::min(max, (uint64_t)test_text.size());
        auto ans = read_file("./data/test_text_ans.txt");

        phase();

		int k = 3;
		int idx = 0;
		for (i = 0; i < max; ++i) {
            auto embed_start = std::chrono::high_resolution_clock::now();
            std::vector<float> query_vec = store.get_embedding(test_text[i]);
            auto embed_end = std::chrono::high_resolution_clock::now();
            get_embedding_call_count++;
            total_embedding_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(embed_end - embed_start).count();

            if (query_vec.empty()) {
                std::cerr << "[WARN] Query Index " << i << ": Failed to get embedding. Skipping query." << std::endl;
                nr_tests += k;
                idx += k;
                continue;
            }

            std::vector<std::pair<uint64_t, std::string>> baseline_results;
            auto baseline_start = std::chrono::high_resolution_clock::now();
            baseline_results = store.search_knn(query_vec, k);
            auto baseline_end = std::chrono::high_resolution_clock::now();
            search_knn_vec_call_count++;
            total_knn_search_only_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(baseline_end - baseline_start).count();

            std::vector<std::pair<uint64_t, std::string>> hnsw_results;
            auto hnsw_start = std::chrono::high_resolution_clock::now();
			hnsw_results = store.search_knn_hnsw(query_vec, k);
            auto hnsw_end = std::chrono::high_resolution_clock::now();
            search_hnsw_vec_call_count++;
            total_hnsw_search_only_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(hnsw_end - hnsw_start).count();

            if (!baseline_results.empty()) {
                std::set<uint64_t> baseline_keys;
                for(const auto& p : baseline_results) {
                    baseline_keys.insert(p.first);
                }
                total_ground_truth_results += baseline_keys.size();

                uint64_t current_matches = 0;
                for(const auto& p : hnsw_results) {
                    if (baseline_keys.count(p.first)) {
                        current_matches++;
                    }
                }
                total_matches_found += current_matches;

            } else {
                 total_ground_truth_results += k;
            }

            int result_index_in_query = 0;
			for (auto j : hnsw_results) {
				if (idx >= ans.size()) {
					 std::cerr << "[WARN] Query Index " << i << " (vs ans.txt): Trying to access ans[" << idx << "] out of bounds (size=" << ans.size() << "). Too many results?" << std::endl;
                     nr_tests++;
					 idx++;
					 result_index_in_query++;
					 continue;
				}
				EXPECT(ans[idx], j.second);
				idx++;
				result_index_in_query++;
			}
            if (result_index_in_query < k) {
                 int missing_results = k - result_index_in_query;
                 nr_tests += missing_results;
                 idx += missing_results;
            }
		}
        print_time_analysis();

		auto phase_with_tolerance = [this](double tolerance = 0.03) {
            std::cout << "\nCorrectness Check (vs ans.txt): ";
			std::cout << nr_passed_tests << "/" << nr_tests << " ";
			double pass_rate = (nr_tests == 0) ? 0.0 : static_cast<double>(nr_passed_tests) / nr_tests;
			bool passed_with_tolerance = pass_rate >= (1.0 - tolerance);
			if (passed_with_tolerance) {
				std::cout << "[PASS]" << std::endl;
			} else {
                std::cout << "[FAIL] (Rate: " << pass_rate * 100 << "%)" << std::endl;
                std::cout << "  Recommended Rate > 85%." << std::endl;
			}
			std::cout.flush();
		};
		phase_with_tolerance(0.15);
	}

    void print_time_analysis() {
        double put_s = static_cast<double>(put_duration_ns) / 1e9;
        double get_s = static_cast<double>(get_duration_ns) / 1e9;
        double embed_s = static_cast<double>(total_embedding_ns) / 1e9;
        double hnsw_search_s = static_cast<double>(total_hnsw_search_only_ns) / 1e9;
        double knn_search_s = static_cast<double>(total_knn_search_only_ns) / 1e9;

        double put_avg_ms = (put_call_count > 0) ? (static_cast<double>(put_duration_ns) / put_call_count / 1e6) : 0.0;
        double get_avg_us = (get_call_count > 0) ? (static_cast<double>(get_duration_ns) / get_call_count / 1e3) : 0.0;
        double embed_avg_ms = (get_embedding_call_count > 0) ? (static_cast<double>(total_embedding_ns) / get_embedding_call_count / 1e6) : 0.0;
        double search_hnsw_avg_us = (search_hnsw_vec_call_count > 0) ? (static_cast<double>(total_hnsw_search_only_ns) / search_hnsw_vec_call_count / 1e3) : 0.0;
        double search_knn_avg_ms = (search_knn_vec_call_count > 0) ? (static_cast<double>(total_knn_search_only_ns) / search_knn_vec_call_count / 1e6) : 0.0;

        double speedup = (hnsw_search_s > 1e-9) ? (knn_search_s / hnsw_search_s) : 0.0;
        double accept_rate = (total_ground_truth_results > 0) ? (static_cast<double>(total_matches_found) / total_ground_truth_results * 100.0) : 0.0;

        std::cout << "\n[Time and Accuracy Analysis]" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "1. Put operations:    " << put_s << "s (" << put_call_count << " calls, avg " << std::fixed << std::setprecision(3) << put_avg_ms << " ms/call)" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "2. Get operations:    " << get_s << "s (" << get_call_count << " calls, avg " << std::fixed << std::setprecision(3) << get_avg_us << " us/call)" << std::endl;

        std::cout << "3. Embedding time:    " << embed_s << "s (" << get_embedding_call_count << " calls, avg " << std::fixed << std::setprecision(3) << embed_avg_ms << " ms/call)" << std::endl;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "4. Baseline KNN search (exc. embedding): " << knn_search_s << "s (" << search_knn_vec_call_count << " calls, avg " << std::fixed << std::setprecision(3) << search_knn_avg_ms << " ms/call)" << std::endl;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "5. HNSW KNN search   (exc. embedding): " << hnsw_search_s << "s (" << search_hnsw_vec_call_count << " calls, avg " << std::fixed << std::setprecision(3) << search_hnsw_avg_us << " us/call";
        if (speedup > 0.0) {
             std::cout << ", Speedup: " << std::fixed << std::setprecision(2) << speedup << "x vs baseline)";
        } else {
            std::cout << ")";
        }
         std::cout << std::endl;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "6. Accept Rate (vs Phase 2 baseline): " << accept_rate << "% (" << total_matches_found << "/" << total_ground_truth_results << " matches)" << std::endl;

        // Print HNSW parameters using getter methods
        std::cout << "HNSW Parameters: M = " << store.get_hnsw_m() << ", efConstruction = " << store.get_hnsw_ef_construction() << std::endl;

        std::cout << "\n[Note:] Internal HNSW function call counts require KVStore modification." << std::endl;
        std::cout << std::endl;
        std::cout.unsetf(std::ios_base::floatfield);
    }

public:
    CorrectnessTest(const std::string &dir, bool v = true) : Test(dir, v) {}

    void start_test(void *args = NULL) override {
        std::cout << "===========================" << std::endl;
        std::cout << "KVStore Correctness Test & Performance Analysis" << std::endl;
        
        store.reset();
        std::cout << "[Text Test]" << std::endl;
        text_test(120);
    }
};

int main(int argc, char *argv[]) {
    bool verbose = (argc == 2 && std::string(argv[1]) == "-v");

    std::cout << "Usage: " << argv[0] << " [-v]" << std::endl;
    std::cout << "  -v: print extra info for failed tests [currently ";
    std::cout << (verbose ? "ON" : "OFF") << "]" << std::endl;
    std::cout << std::endl;
    std::cout.flush();

    CorrectnessTest test("./data", verbose);

    test.start_test();

    return 0;
}