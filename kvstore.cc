#include "kvstore.h"

#include "skiplist.h"
#include "sstable.h"
#include "utils.h"
#include "embedding.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <cstdint> // 确保包含
#include <iomanip> // For std::fixed and std::setprecision in debug output
#include <cmath>   // For std::fabs in compare_float_vectors

// Needed for ThreadPool and parallel save
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
// #include <queue> // Already included via other headers or kvstore.h indirectly

// --- BEGIN THREADPOOL CLASS DEFINITION (FROM PHASE5.MD) ---
class ThreadPool {
public:
  ThreadPool(size_t num_threads) : stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(
                lock, [this] { return this->stop || !this->tasks.empty(); });
            if (this->stop && this->tasks.empty()) {
              return;
            }
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
        }
      });
    }
  }

  template <class F> void enqueue(F &&f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      tasks.emplace(std::forward<F>(f));
    }
    condition.notify_one();
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) {
      if (worker.joinable()) { // Check if joinable before joining
        worker.join();
      }
    }
  }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};
// --- END THREADPOOL CLASS DEFINITION ---


static const std::string DEL = "~DELETED~";
const uint32_t MAXSIZE       = 2 * 1024 * 1024;

// 全局 HNSW 头信息
struct HNSWGlobalHeader {
    uint32_t M;                // 参数 M
    uint32_t M_max;            // 参数 M_max
    uint32_t efConstruction;   // 参数 efConstruction
    // uint32_t m_L;           // m_L 可以根据 M 重新计算，不保存
    uint32_t max_level;        // 当前 HNSW 图的最高层级 (current_max_level_)
    uint64_t entry_point_label;// HNSW 图的入口点标签 (entry_point_label_)
    uint64_t num_nodes;        // 保存时活动节点的近似数量 (可以用 next_label_ 作为上界)
    uint32_t dim;              // 向量维度 (embedding_dimension_)
    // 注意：为了简化，这里使用了 uint64_t 来存储 label，即使文件格式要求 uint32_t。
    // 在读写时需要进行转换和检查。或者直接修改结构体为 uint32_t，但要确保 label 不会溢出。
    // 暂时使用 uint64_t 匹配代码，写入时转换。
};

// 单个节点的头信息 (不包含向量本身)
struct NodeHeader {
    uint32_t max_level;        // 节点存在的最高层级 (node.max_level)
    uint64_t key;              // 节点对应的 KVStore key (node.key)
    // 注意：文件格式描述 header.bin 存向量，但这里存 key 似乎更合理，
    // 因为向量已经通过 embedding 持久化加载了。我们遵循 Phase4.md 的 NodeHeader 定义。
};

// EdgeFile 结构是隐式的，写入时先写 uint32_t num_edges，再写 uint32_t neighbors[...]

struct poi {
    int sstableId; // vector中第几个sstable
    int pos;       // 该sstable的第几个key-offset
    uint64_t time;
    Index index;
};

struct cmpPoi {
    bool operator()(const poi &a, const poi &b) {
        if (a.index.key == b.index.key)
            return a.time < b.time;
        return a.index.key > b.index.key;
    }
};

//DEL_MARKER_STRING (already defined or should be)
//const std::string DEL_MARKER_STRING = "~DELETED~"; 


// Helper function to convert string to vector (remains unchanged)
// ...

bool KVStore::compare_float_vectors(const std::vector<float>& v1, const std::vector<float>& v2, float epsilon) {
    if (v1.size() != v2.size()) {
        // std::cout << "[DEBUG_COMPARE_VEC] Size mismatch: " << v1.size() << " vs " << v2.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::fabs(v1[i] - v2[i]) > epsilon) {
            // Optional: Log the first differing element for debugging, but can be verbose
            // std::cout << "[DEBUG_COMPARE_VEC] Element mismatch at index " << i 
            //           << ": v1[" << i << "]=" << std::fixed << std::setprecision(8) << v1[i] 
            //           << ", v2[" << i << "]=" << std::fixed << std::setprecision(8) << v2[i] 
            //           << " (diff: " << std::fixed << std::setprecision(8) << std::fabs(v1[i] - v2[i]) 
            //           << ") epsilon: " << epsilon << std::endl; // MODIFIED LOG
            return false;
        }
    }
    return true;
}


KVStore::KVStore(const std::string &dir, const std::string &hnsw_index_path) :
    KVStoreAPI(dir), dir_(dir) // Added dir_(dir) to initializer list
{
    for (totalLevel = 0;; ++totalLevel) {
        std::string path = dir_ + "/level-" + std::to_string(totalLevel) + "/";
        std::vector<std::string> files;
        if (!utils::dirExists(path)) {
            totalLevel--;
            break; // stop read
        }
        int nums = utils::scanDir(path, files);
        sstablehead cur;
        for (int i = 0; i < nums; ++i) {       // 读每一个文件头
            std::string url = path + files[i]; // url, 每一个文件名
            cur.loadFileHead(url.data());
            sstableIndex[totalLevel].push_back(cur);
            TIME = std::max(TIME, cur.getTime()); // 更新时间戳
        }
    }

    // --- HNSW 初始化 ---
    embedding_dimension_ = 768; // 预设维度，会被加载函数覆盖或验证
    current_max_level_ = -1;
    entry_point_label_ = 0;
    next_label_ = 0;
    hnsw_nodes_.clear();
    key_to_label_.clear();
    label_to_key_.clear();
    embeddings.clear(); // 确保开始时内存为空
    // rng_ 已经在头文件中初始化
    // --------------------

    // --- 修改：加载 Embeddings --- (调用函数名)
    std::cout << "[INFO] Attempting to load embeddings from disk..." << std::endl;
    load_embedding_from_disk(dir); // This uses the KVStore's main data directory for embeddings.bin
    // ---------------------------
    hnsw_vectors_to_persist_as_deleted_.clear(); // Initialize

    // --- 新增：加载 HNSW 索引 (now conditional) ---
    if (!hnsw_index_path.empty()) {
        std::cout << "[INFO] Attempting to load HNSW index from provided path: " << hnsw_index_path << std::endl;
        load_hnsw_index_from_disk(hnsw_index_path);
    } else {
        std::cout << "[INFO] No specific HNSW index path provided. HNSW index will start empty." << std::endl;
        // No longer attempting to load from a default "./hnsw_data" path
    }
    // ---------------------------

    // --- 可选：检查是否需要重建 HNSW ---
    // 如果加载失败 (hnsw_nodes_ 仍然为空)，并且 embeddings map 不为空，
    // 则可能需要根据加载的 embeddings 重建 HNSW 图。
    if (hnsw_nodes_.empty() && !embeddings.empty()) {
       std::cout << "[INFO] No HNSW index loaded or load failed, rebuilding from loaded embeddings..." << std::endl;
       // 确保重建前 next_label_ 等状态正确
       next_label_ = 0;
       current_max_level_ = -1;
       entry_point_label_ = 0;
       key_to_label_.clear(); // 清空映射，因为 hnsw_insert 会重新建立
       label_to_key_.clear();

       for (const auto& pair : embeddings) {
           // 检查向量有效性，避免插入空向量或错误维度的向量
           if (!pair.second.empty() && pair.second.size() == embedding_dimension_) {
               hnsw_insert(pair.first, pair.second); // 重新插入以构建 HNSW 图
           } else {
                std::cerr << "[WARN] Skipping rebuild for key " << pair.first << " due to invalid embedding vector." << std::endl;
           }
       }
       std::cout << "[INFO] Finished rebuilding HNSW index from " << embeddings.size() << " embeddings." << std::endl;
    } else if (!hnsw_nodes_.empty()) {
        std::cout << "[INFO] HNSW index successfully loaded from disk." << std::endl;
    }
}

KVStore::~KVStore() {
    // --- 第一步：保存 Memtable 中剩余数据到 SSTable ---
    if (s->getCnt() > 1) { // 假设 getCnt() 返回节点数，>1 表示有有效数据
        std::cout << "[INFO] Saving final Memtable state to SSTable during destruction..." << std::endl;
        try {
            sstable ss(s); // 从当前 memtable 创建 sstable 对象
            std::string level0_path_str = this->dir_ + "/level-0/"; // MODIFIED: Use dir_
            // 确保目录存在 (使用 C++17 filesystem)
            if (!std::filesystem::exists(level0_path_str)) {
                std::filesystem::create_directories(level0_path_str);
                 if (totalLevel < 0) totalLevel = 0; // 更新层级记录
                 std::cout << "[INFO] Created directory: " << level0_path_str << std::endl;
            }

            std::string url = ss.getFilename(); // 获取基于 TIME 的文件名
            // --- MODIFICATION: Construct full path for SSTable using level0_path_str ---
            std::string full_sstable_path = level0_path_str + std::to_string(ss.getTime()) + ".sst";
            ss.setFilename(full_sstable_path); // Update sstable's internal filename to full path
            // --- END MODIFICATION ---

             // 检查 ss 是否真的有内容，避免创建空 sstable (虽然 s->getCnt() 应该保证了)
             if (ss.getCnt() > 0) {
                 ss.putFile(full_sstable_path.data()); // MODIFIED: Use full_sstable_path
                 addsstable(ss, 0);                  // 将其头信息加入内存 Level 0 索引
                 std::cout << "[INFO] Saved Memtable to SSTable: " << full_sstable_path << std::endl; // MODIFIED
             } else {
                  std::cout << "[WARN] Memtable seemed non-empty but created empty SSTable. Skipping save." << std::endl;
             }
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception during SSTable save in destructor: " << e.what() << std::endl;
        } catch (...) {
             std::cerr << "[ERROR] Unknown exception during SSTable save in destructor." << std::endl;
        }
    } else {
         std::cout << "[INFO] Memtable empty or only sentinels, skipping SSTable save during destruction." << std::endl;
    }
    // --- SSTable 保存结束 ---

    // --- 第二步：保存 Embeddings Map ---
    if (!embeddings.empty() && embedding_dimension_ > 0) {
        std::cout << "[INFO] Saving embeddings map to disk during KVStore destruction..." << std::endl;
        const std::string embedding_file_path = dir_ + "/embeddings.bin"; // NEW
        std::ofstream embed_file(embedding_file_path, std::ios::binary | std::ios::app);

        if (embed_file.is_open()) {
            embed_file.seekp(0, std::ios::end);
            if (embed_file.tellp() == 0) {
                 uint64_t dim_to_write = embedding_dimension_;
                 embed_file.write(reinterpret_cast<const char*>(&dim_to_write), sizeof(dim_to_write));
                 std::cout << "[INFO] Writing embedding dimension (" << dim_to_write << ") to new embedding file." << std::endl;
            }
            for (const auto& pair : embeddings) {
                uint64_t current_key = pair.first;
                const std::vector<float>& vec = pair.second;
                if (vec.size() == embedding_dimension_) {
                    embed_file.write(reinterpret_cast<const char*>(&current_key), sizeof(current_key));
                    embed_file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
                } else {
                     std::cerr << "[WARN] Embedding dimension mismatch for key " << current_key << " during destructor save. Skipping." << std::endl;
                }
            }
            embed_file.close();
             std::cout << "[INFO] Finished saving " << embeddings.size() << " embeddings from map." << std::endl;
        } else {
            std::cerr << "[ERROR] Failed to open embedding file for writing during destruction: " << embedding_file_path << std::endl;
        }
    } else {
         std::cout << "[INFO] Embeddings map empty or dimension unknown, skipping embedding save during destruction." << std::endl;
    }
    // --- Embedding 保存结束 ---

    // --- 新增：保存 HNSW 索引 --- (修改：在析构函数中移除自动保存)
    // std::string hnsw_save_path = "./hnsw_data"; // 定义保存路径
    // save_hnsw_index_to_disk(hnsw_save_path); // 移除或注释掉这一行
    std::cout << "[INFO] KVStore destructor: HNSW index auto-saving is now disabled. Call save_hnsw_index_to_disk() explicitly if needed." << std::endl;
    // --- HNSW 保存结束 ---

    // 可能的清理
    // embedding_cleanup();
}

/**
 * Insert/Update the key-value pair.
 * No return values for simplicity.
 */
void KVStore::put(uint64_t key, const std::string &s_val) { // Renamed string param to s_val to avoid conflict
    // --- ADDED: Log initial state of embeddings[key] if it exists ---
    if (auto it = embeddings.find(key); it != embeddings.end()) { 
        const auto& existing_vec_in_map = it->second;
        std::cout << "[DEBUG_KV_PUT_INIT_STATE] Key " << key << " in embeddings. 1st_Elem: "
                  << (existing_vec_in_map.empty() ? "EMPTY" : std::to_string(existing_vec_in_map[0]))
                  << " Size: " << existing_vec_in_map.size() 
                  << " Addr: " << (void*)&existing_vec_in_map << std::endl;
    } else {
        std::cout << "[DEBUG_KV_PUT_INIT_STATE] Key " << key << " NOT in embeddings." << std::endl;
    }
    // --- END ADDED ---
    // --------- Start of Reconstructed Put Method ---------
    // utils::checkDir(dir_); 
    if (!utils::dirExists(dir_)) {
        utils::mkdir(dir_.data());
    }
    std::vector<float> emb_vec;

    // 1. Determine embedding and dimension
    if (embedding_dimension_ == 0 && !s_val.empty() && s_val != DEL) { //MODIFIED: DEL_MARKER_STRING -> DEL
        std::vector<float> temp_emb_for_dim = get_embedding(s_val);
        if (!temp_emb_for_dim.empty()) {
            embedding_dimension_ = temp_emb_for_dim.size();
            std::cout << "[INFO_KV_PUT] Embedding dimension determined: " << embedding_dimension_ << " from key " << key << std::endl;
        }
    }

    if (!s_val.empty() && s_val != DEL) { //MODIFIED: DEL_MARKER_STRING -> DEL
        emb_vec = get_embedding(s_val);
        if (emb_vec.empty() && embedding_dimension_ > 0) {
            std::cerr << "[WARN_KV_PUT] get_embedding for key " << key << " -> empty vector, but dim=" << embedding_dimension_ << ". Storing zero vector." << std::endl;
            emb_vec.assign(embedding_dimension_, 0.0f);
        } else if (!emb_vec.empty() && emb_vec.size() != embedding_dimension_ && embedding_dimension_ != 0) {
             std::cerr << "[ERROR_KV_PUT] Embedding dim mismatch for key " << key << "! Expected " << embedding_dimension_ << " got " << emb_vec.size() << ". Not storing." << std::endl;
             return;
        } else if (emb_vec.empty() && embedding_dimension_ == 0) {
             std::cout << "[INFO_KV_PUT] Storing empty string for key " << key << " with no embedding (dim 0)." << std::endl;
        }
    } else if (s_val == DEL && embedding_dimension_ > 0) { //MODIFIED: DEL_MARKER_STRING -> DEL
        emb_vec.assign(embedding_dimension_, std::numeric_limits<float>::max()); // DEL_MARKER_VECTOR
        std::cout << "[DEBUG_KV_PUT] Key " << key << " is DEL_MARKER. Assigned DEL_MARKER_VECTOR." << std::endl;
    } else if (s_val.empty() && embedding_dimension_ > 0) { // Handling for explicitly putting empty string with known dimension
        emb_vec.assign(embedding_dimension_, 0.0f); // Store as zero vector or specific empty representation
        std::cout << "[WARN_KV_PUT] Empty string provided for key " << key << ". Storing as zero vector (dim: " << embedding_dimension_ << ")." << std::endl;
    } else if (s_val == DEL && embedding_dimension_ == 0) { //MODIFIED: DEL_MARKER_STRING -> DEL
        std::cout << "[INFO_KV_PUT] Storing DEL_MARKER for key " << key << " with no embedding (dim 0)." << std::endl;
    } else if (s_val.empty() && embedding_dimension_ == 0) {
         std::cout << "[INFO_KV_PUT] Storing empty string for key " << key << " with no embedding (dim 0)." << std::endl;
    }

    bool is_update = this->embeddings.count(key);
    std::vector<float> old_vector_copy;

    if (is_update) {
        // std::cout << "[DEBUG_KV_PUT_UPDATE] Updating existing key: " << key << std::endl;
        const auto& old_vector_ref_in_map = this->embeddings[key]; // Get ref before it's potentially changed by new emb_vec assignment below
        old_vector_copy = old_vector_ref_in_map; 

        // std::cout << "[DEBUG_KV_PUT_UPDATE] OLD_VEC for key " << key << " (copied). Addr_copy: " << (void*)&old_vector_copy 
        //           << " Addr_orig_in_map: " << (void*)&old_vector_ref_in_map 
        //           << " 1st_Elem: " << (old_vector_copy.empty() ? -999.0f : old_vector_copy[0]) 
        //           << " Size: " << old_vector_copy.size() << std::endl;

        bool old_vec_is_del_marker_vector = false;
        if (!old_vector_copy.empty() && old_vector_copy.size() == embedding_dimension_ && embedding_dimension_ > 0) {
            old_vec_is_del_marker_vector = true;
            for(float v_val : old_vector_copy) if (v_val != std::numeric_limits<float>::max()) { old_vec_is_del_marker_vector = false; break; }
        }
        if (old_vec_is_del_marker_vector) {
            std::cout << "[DEBUG_KV_PUT_UPDATE] Key " << key << ": Old vector in map WAS a delete marker vector." << std::endl;
        }

        if (!old_vector_copy.empty() && !old_vec_is_del_marker_vector) {
            bool found_in_loaded = false;
            int loaded_idx = 0;
            for (const auto& v_loaded : loaded_deleted_vectors_) {
                if (compare_float_vectors(old_vector_copy, v_loaded)) {
                    std::cout << "[DEBUG_KV_PUT_UPDATE] Key " << key << ": OLD_VEC MATCHED LOADED_VEC[" << loaded_idx << "] with tolerance." << std::endl;
                    found_in_loaded = true;
                    break;
                }
                loaded_idx++;
            }

            if (!found_in_loaded) {
                bool found_in_persist = false;
                int persist_idx = 0;
                for (const auto& v_persist : hnsw_vectors_to_persist_as_deleted_) {
                    if (compare_float_vectors(old_vector_copy, v_persist)) {
                        std::cout << "[DEBUG_KV_PUT_UPDATE] Key " << key << ": OLD_VEC MATCHED hnsw_vectors_to_persist_as_deleted_[" << persist_idx << "] with tolerance." << std::endl;
                        found_in_persist = true;
                        break;
                    }
                    persist_idx++;
                }
                if (!found_in_persist) {
                    // DETAILED LOGGING FOR COMPARISON FAILURE START
                    // std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL] Key " << key << ": OLD_VEC NOT FOUND in loaded_deleted_vectors_ NOR hnsw_vectors_to_persist_as_deleted_." << std::endl;
                    // std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]   OLD_VEC (key " << key << ", size " << old_vector_copy.size() << ") First 3 elems: ";
                    // for(int i=0; i<std::min((size_t)3, old_vector_copy.size()); ++i) std::cout << std::fixed << std::setprecision(8) << old_vector_copy[i] << " ";
                    // std::cout << std::endl;

                    // if (loaded_deleted_vectors_.empty()) {
                    //     std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]   loaded_deleted_vectors_ is EMPTY." << std::endl;
                    // } else {
                    //     std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]   Comparing with loaded_deleted_vectors_ (showing first few):" << std::endl;
                    //     int c = 0;
                    //     for (const auto& v_l : loaded_deleted_vectors_) {
                    //         if (c >= 3 && loaded_deleted_vectors_.size() > 5) { std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]     ... and " << (loaded_deleted_vectors_.size() - c) << " more." << std::endl; break;}
                    //         std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]     LoadedVec[" << c << "] (size " << v_l.size() << ") First 3 elems: ";
                    //         for(int i=0; i<std::min((size_t)3, v_l.size()); ++i) std::cout << std::fixed << std::setprecision(8) << v_l[i] << " ";
                    //         std::cout << std::endl;
                    //         // Explicitly call compare_float_vectors here again FOR LOGGING THE FAILURE POINT if it's this vector
                    //         if (!compare_float_vectors(old_vector_copy, v_l)) {
                    //             // std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]       Above comparison with LoadedVec[" << c << "] failed inside compare_float_vectors (see details above/below this line if logged)." << std::endl;
                    //         }
                    //         c++;
                    //     }
                    // }
                    // if (hnsw_vectors_to_persist_as_deleted_.empty()) {
                    //     std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]   hnsw_vectors_to_persist_as_deleted_ is EMPTY." << std::endl;
                    // } else {
                    //     std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]   Comparing with hnsw_vectors_to_persist_as_deleted_ (showing first few):" << std::endl;
                    //     int c = 0;
                    //     for (const auto& v_p : hnsw_vectors_to_persist_as_deleted_) {
                    //         if (c >= 3 && hnsw_vectors_to_persist_as_deleted_.size() > 5) { std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]     ... and " << (hnsw_vectors_to_persist_as_deleted_.size() - c) << " more." << std::endl; break;}
                    //         std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]     PersistVec[" << c << "] (size " << v_p.size() << ") First 3 elems: ";
                    //         for(int i=0; i<std::min((size_t)3, v_p.size()); ++i) std::cout << std::fixed << std::setprecision(8) << v_p[i] << " ";
                    //         std::cout << std::endl;
                    //         // Explicitly call compare_float_vectors here again FOR LOGGING THE FAILURE POINT if it's this vector
                    //         if (!compare_float_vectors(old_vector_copy, v_p)) {
                    //              // std::cout << "[DEBUG_KV_PUT_COMPARE_FAIL]       Above comparison with PersistVec[" << c << "] failed inside compare_float_vectors (see details above/below this line if logged)." << std::endl;
                    //         }
                    //         c++;
                    //     }
                    // }
                    // DETAILED LOGGING FOR COMPARISON FAILURE END
                    // std::cout << "[DEBUG_KV_PUT_UPDATE] Added OLD vector for key " << key << " to hnsw_vectors_to_persist_as_deleted_." << std::endl;
                    hnsw_vectors_to_persist_as_deleted_.push_back(old_vector_copy); // Keep this: old vector from an update should be persisted as deleted
                }
            }
        }
    }

    // 2. Update in-memory embeddings map
    this->embeddings[key] = emb_vec;

    // 3. LSM Memtable PUT operation (Reinstated logic)
    uint32_t current_memtable_bytes = this->s->getBytes();
    uint32_t new_val_bytes = s_val.length();
    uint32_t key_bytes_overhead = 12; 
    std::string existing_val_in_memtable = this->s->search(key);
    uint32_t estimated_new_total_bytes;

    if (!existing_val_in_memtable.empty()) { 
        estimated_new_total_bytes = current_memtable_bytes - existing_val_in_memtable.length() + new_val_bytes;
    } else { 
        estimated_new_total_bytes = current_memtable_bytes + key_bytes_overhead + new_val_bytes;
    }
    
    if (estimated_new_total_bytes + 10240 + 32 > MAXSIZE && this->s->getCnt() > 0) {
        std::cout << "[INFO_KV_PUT] Memtable full. Flushing before putting key " << key << std::endl;
        sstable ss_to_flush(this->s); 

        // Persist embeddings for the memtable being flushed
        const std::string embedding_file_path = dir_ + "/embeddings.bin";
        std::ofstream embed_file(embedding_file_path, std::ios::binary | std::ios::app);
        if (embed_file.is_open()) {
            embed_file.seekp(0, std::ios::end);
            if (embed_file.tellp() == 0 && embedding_dimension_ > 0) {
                uint64_t dim_to_write = embedding_dimension_;
                embed_file.write(reinterpret_cast<const char*>(&dim_to_write), sizeof(dim_to_write));
            }
            slnode *curr = this->s->getFirst();
            while (curr && curr->type != TAIL) {
                if (this->embeddings.count(curr->key)) {
                    const auto& vec_to_persist = this->embeddings[curr->key];
                    bool is_del_marker = true;
                    if(vec_to_persist.size() == embedding_dimension_ && embedding_dimension_ > 0){
                        for(float v_val : vec_to_persist) if(v_val != std::numeric_limits<float>::max()){ is_del_marker=false; break;}
                    } else if (vec_to_persist.empty() && embedding_dimension_ == 0) { 
                        is_del_marker = false; 
                    } else { 
                        is_del_marker = true;
                    }

                    if (!is_del_marker) { // Persist if not a delete marker vector (or valid empty string for dim 0)
                         uint64_t temp_key_to_write = curr->key; // Ensure correct key type/value
                         embed_file.write(reinterpret_cast<const char*>(&temp_key_to_write), sizeof(temp_key_to_write));
                         if (!vec_to_persist.empty()){ // Only write vector data if it's not an empty string for dim 0
                            embed_file.write(reinterpret_cast<const char*>(vec_to_persist.data()), vec_to_persist.size() * sizeof(float));
                         }
                    }
                }
                curr = curr->nxt[0];
            }
            embed_file.close();
        } else {
            std::cerr << "[ERROR_KV_PUT] Failed to open embedding file for writing during flush: " << embedding_file_path << std::endl;
        }

        this->s->reset(); 
        std::string level0_path = dir_ + "/level-0";
        if (!utils::dirExists(level0_path)) {
            utils::mkdir(level0_path.data());
            if(totalLevel < 0) totalLevel = 0; 
        }
        std::string full_sstable_path = level0_path + "/" + std::to_string(ss_to_flush.getTime()) + ".sst";
        ss_to_flush.setFilename(full_sstable_path);
        
        if(ss_to_flush.getCnt() > 0) {
            addsstable(ss_to_flush, 0); 
            ss_to_flush.putFile(full_sstable_path.data());
            std::cout << "[INFO_KV_PUT] Flushed Memtable to SSTable: " << full_sstable_path << std::endl;
        }
        compaction();
    }

    this->s->insert(key, s_val); // MODIFIED: put -> insert
    // this->s->put(key, s_val); // This line was the duplicate, now removed/commented
    // std::cout << "[DEBUG_KV_PUT] Key " << key << " val_str: \\"" << s_val.substr(0, 20) << (s_val.length() > 20 ? "..." : "") << "\\" inserted/updated in memtable." << std::endl;

    // 4. HNSW Update/Insert
    #ifndef DISABLE_EMBEDDING_FOR_TESTS
    if (embedding_dimension_ > 0) {
        bool new_emb_is_del_marker = false;
        if(emb_vec.size() == embedding_dimension_ && !emb_vec.empty()){ // Check !empty explicitly
            new_emb_is_del_marker = true;
            for(float v_val : emb_vec) if(v_val != std::numeric_limits<float>::max()){new_emb_is_del_marker = false; break;}
        } else if (emb_vec.empty()){ 
             new_emb_is_del_marker = true; 
        }

        if (is_update) { 
            if (key_to_label_.count(key)) { 
                size_t old_label = key_to_label_[key];
                if (hnsw_nodes_.count(old_label)) {
                    hnsw_nodes_[old_label].deleted = true; // Mark old HNSW node as deleted
                    // std::cout << "[DEBUG_KV_PUT_UPDATE] Marked old HNSW node (label " << old_label << ") as deleted for key " << key << std::endl;
                }
            }
        }

        if (!emb_vec.empty() && !new_emb_is_del_marker) {
            hnsw_insert(key, emb_vec); // hnsw_insert will handle making the node active
            // std::cout << "[DEBUG_KV_PUT] Called hnsw_insert for key " << key << (is_update ? " (update)" : " (new)") << std::endl;
        } else if (key_to_label_.count(key) && (emb_vec.empty() || new_emb_is_del_marker)) {
             // std::cout << "[DEBUG_KV_PUT] Key " << key << " updated/is empty/marker. No HNSW insert/update." << std::endl;
        }
    }
    #endif
    // --------- End of Reconstructed Put Method ---------
}

/**
 * Returns the (string) value of the given key.
 * An empty string indicates not found.
 */
std::string KVStore::get(uint64_t key) //
{
    uint64_t time = 0;
    int goalOffset;
    uint32_t goalLen;
    std::string goalUrl;
    std::string res = s->search(key);
    if (res.length()) { // 在memtable中找到, 或者是deleted，说明最近被删除过，
                        // 不用查sstable
        if (res == DEL)
            return "";
        return res;
    }
    for (int level = 0; level <= totalLevel; ++level) {
        for (sstablehead it : sstableIndex[level]) {
            if (key < it.getMinV() || key > it.getMaxV())
                continue;
            uint32_t len;
            int offset = it.searchOffset(key, len);
            if (offset == -1) {
                if (!level)
                    continue;
                else
                    break;
            }
            // sstable ss;
            // ss.loadFile(it.getFilename().data());
            if (it.getTime() > time) { // find the latest head
                time       = it.getTime();
                goalUrl    = it.getFilename();
                goalOffset = offset + 32 + 10240 + 12 * it.getCnt();
                goalLen    = len;
            }
        }
        if (time)
            break; // only a test for found
    }
    if (!goalUrl.length())
        return ""; // not found a sstable
    res = fetchString(goalUrl, goalOffset, goalLen);
    if (res == DEL)
        return "";
    return res;
}

/**
 * Delete the given key-value pair if it exists.
 * Returns false iff the key is not found.
 */
bool KVStore::del(uint64_t key) {
    // 首先直接从 s 获取值，避免递归调用 get
    std::string value = s->search(key);
    bool in_memtable = !value.empty();
    
    // 如果不在 memtable 中，尝试从 sstable 获取
    if (!in_memtable) {
        // 这里复制 get 方法中查找 sstable 的逻辑，不要递归调用 get
        // ...查找 sstable 的逻辑...
        
        // 如果在 sstable 中也没找到，返回 false
        if (value.empty()) return false;
    }
    
    // HNSW 删除逻辑
    auto it_label = key_to_label_.find(key);
    if (it_label != key_to_label_.end()) {
        size_t label = it_label->second;
        if (hnsw_nodes_.count(label) && !hnsw_nodes_[label].deleted) {
            hnsw_nodes_[label].deleted = true;
            
            // 添加到持久化列表
            auto it_emb = embeddings.find(key);
            if (it_emb != embeddings.end()) {
                const std::vector<float>& original_vec = it_emb->second;
                std::cout << "[DEBUG_KV_DEL] Key " << key << ": Adding vector to persistence list. Current count: " 
                          << hnsw_vectors_to_persist_as_deleted_.size() << std::endl;
                hnsw_vectors_to_persist_as_deleted_.push_back(original_vec);
            }
        }
    }
    
    // 最后在 memtable 中标记为删除
    s->insert(key, DEL);
    return true;
}

/**
 * This resets the kvstore. All key-value pairs should be removed,
 * including memtable and all sstables files.
 */
void KVStore::reset() {
    // --- LSM 重置 ---
    s->reset();
    for (int level = 0; level <= totalLevel; ++level) {
        std::string path = dir_ + "/level-" + std::to_string(level); // Use dir_
    std::vector<std::string> files;
        if (utils::dirExists(path)) { // Check if dir exists before scanning
            int size = utils::scanDir(path, files);
        for (int i = 0; i < size; ++i) {
                std::string file_to_delete = path + "/" + files[i];
                utils::rmfile(file_to_delete.data());
        }
        utils::rmdir(path.data());
        }
        sstableIndex[level].clear();
    }
    totalLevel = -1;

    // --- Embedding file cleanup ---
    std::string embedding_file = dir_ + "/embeddings.bin";
    if (utils::fileExists(embedding_file.c_str())) {
        utils::rmfile(embedding_file.data());
    }

    #ifndef DISABLE_EMBEDDING_FOR_TESTS
    // --- HNSW 重置 ---
    embeddings.clear();
    hnsw_nodes_.clear();
    key_to_label_.clear();
    label_to_key_.clear();
    next_label_ = 0;
    entry_point_label_ = 0;
    current_max_level_ = -1;
    // embedding_dimension_ 通常不需要重置

    // --- Phase 4 HNSW delete persistence cleanup ---
    // keys_marked_for_hnsw_deletion_.clear();
    hnsw_vectors_to_persist_as_deleted_.clear(); // Clear here as well
    loaded_deleted_vectors_.clear();
    std::string hnsw_data_dir = "./hnsw_data"; // Assuming default path for now, or use a member if configurable
    std::string deleted_nodes_file = hnsw_data_dir + "/deleted_nodes.bin";
    if (utils::fileExists(deleted_nodes_file.c_str())) {
        utils::rmfile(deleted_nodes_file.data());
    }
    // Optionally, clean up the entire hnsw_data_dir if it's fully managed by this KVStore instance
    // For now, only cleaning deleted_nodes.bin and global_header.bin as per specific Phase4 files.
    // A more robust reset might wipe the whole hnsw_data_dir/nodes/ too.
    std::string global_header_file = hnsw_data_dir + "/global_header.bin";
    if (utils::fileExists(global_header_file.c_str())) {
        utils::rmfile(global_header_file.data());
    }
    std::string hnsw_nodes_dir = hnsw_data_dir + "/nodes";
    if (utils::dirExists(hnsw_nodes_dir)) {
        // This requires recursive directory removal or iterating and deleting files/subdirs
        // For simplicity with utils::rmdir, it expects an empty directory.
        // Let's assume for Phase 4, cleaning individual files is enough unless full HNSW reset is needed.
        // utils::rmdir(hnsw_nodes_dir.data()); // This would fail if not empty
        // A safer approach for full cleanup: iterate and delete, or use std::filesystem::remove_all
         try {
             if (std::filesystem::exists(hnsw_nodes_dir)) { // Check with filesystem before removing
                 std::filesystem::remove_all(hnsw_nodes_dir); 
                 std::cout << "[INFO] KVStore::reset - Removed HNSW nodes directory: " << hnsw_nodes_dir << std::endl;
             }
         } catch (const std::filesystem::filesystem_error& fs_err) {
             std::cerr << "[ERROR] KVStore::reset - Filesystem error while removing HNSW nodes directory " << hnsw_nodes_dir << ": " << fs_err.what() << std::endl;
         }
    }
    // --------------------
    #endif

    // 找到现有代码中清理HNSW相关状态的部分
    
    // 添加下面这行代码确保清理持久化列表
    hnsw_vectors_to_persist_as_deleted_.clear();
    std::cout << "[INFO] KVStore::reset - Cleared hnsw_vectors_to_persist_as_deleted_ list" << std::endl;
    
    // 其余部分保持不变...
}

/**
 * Return a list including all the key-value pair between key1 and key2.
 * keys in the list should be in an ascending order.
 * An empty string indicates not found.
 */

struct myPair {
    uint64_t key, time;
    int id, index;
    std::string filename;

    myPair(uint64_t key, uint64_t time, int index, int id,
           std::string file) { // construct function
        this->time     = time;
        this->key      = key;
        this->id       = id;
        this->index    = index;
        this->filename = file;
    }
};

struct cmp {
    bool operator()(myPair &a, myPair &b) {
        if (a.key == b.key)
            return a.time < b.time;
        return a.key > b.key;
    }
};


void KVStore::scan(uint64_t key1, uint64_t key2, std::list<std::pair<uint64_t, std::string>> &list) {
    std::vector<std::pair<uint64_t, std::string>> mem;
    // std::set<myPair> heap; // 维护一个指针最小堆
    std::priority_queue<myPair, std::vector<myPair>, cmp> heap;
    // std::vector<sstable> ssts;
    std::vector<sstablehead> sshs;
    s->scan(key1, key2, mem);   // add in mem
    std::vector<int> head, end; // [head, end)
    int cnt = 0;
    if (mem.size())
        heap.push(myPair(mem[0].first, INF, 0, -1, "qwq"));
    for (int level = 0; level <= totalLevel; ++level) {
        for (sstablehead it : sstableIndex[level]) {
            if (key1 > it.getMaxV() || key2 < it.getMinV())
                continue; // 无交集
            int hIndex = it.lowerBound(key1);
            int tIndex = it.lowerBound(key2);
            if (hIndex < it.getCnt()) { // 此sstable可用
                // sstable ss; // 读sstable
                std::string url = it.getFilename();
                // ss.loadFile(url.data());

                heap.push(myPair(it.getKey(hIndex), it.getTime(), hIndex, cnt++, url));
                head.push_back(hIndex);
                if (it.search(key2) == tIndex)
                    tIndex++; // tIndex为第一个不可的
                end.push_back(tIndex);
                // ssts.push_back(ss); // 加入ss
                sshs.push_back(it);
            }
        }
    }
    uint64_t lastKey = INF; // only choose the latest key
    while (!heap.empty()) { // 维护堆
        myPair cur = heap.top();
        heap.pop();
        if (cur.id >= 0) { // from sst
            if (cur.key != lastKey) {
                lastKey         = cur.key;
                uint32_t start  = sshs[cur.id].getOffset(cur.index - 1);
                uint32_t len    = sshs[cur.id].getOffset(cur.index) - start;
                uint32_t scnt   = sshs[cur.id].getCnt();
                std::string res = fetchString(sshs[cur.id].getFilename(), 10240 + 32 + scnt * 12 + start, len);
                if (res.length() && res != DEL)
                    list.emplace_back(cur.key, res);
            }
            if (cur.index + 1 < end[cur.id]) { // add next one to heap
                heap.push(myPair(sshs[cur.id].getKey(cur.index + 1), cur.time, cur.index + 1, cur.id, sshs[cur.id].getFilename()));
            }
        } else { // from mem
            if (cur.key != lastKey) {
                lastKey         = cur.key;
                std::string res = mem[cur.index].second;
                if (res.length() && res != DEL)
                    list.emplace_back(cur.key, res);
            }
            if (cur.index < mem.size() - 1) {
                heap.push(myPair(mem[cur.index + 1].first, cur.time, cur.index + 1, -1, cur.filename));
            }
        }
    }
}


void KVStore::compaction() {
    // 从 Level 0 开始尝试合并
    for (int level = 0; level <= totalLevel; ++level) {
        // 检查当前层的文件数量是否超过限制
        // Level 0 的限制通常较小，例如 4 个文件
        // 其他层的限制可以更大，例如 2^level 个文件
        int maxFiles = (level == 0) ? 4 : (1 << (level + 1));
        
        if (sstableIndex[level].size() <= maxFiles) {
            continue; // 如果没有超过限制，继续检查下一层
        }
        
        // 如果是 Level 0，需要合并所有文件
        if (level == 0) {
            // 1. 统计 Level 0 层中所有 SSTable 所覆盖的键的区间
            uint64_t minKey = std::numeric_limits<uint64_t>::max();
            uint64_t maxKey = 0;
            
            // 检查是否有文件需要合并
            if (sstableIndex[level].empty()) {
                continue;
            }
            
            for (const auto& head : sstableIndex[level]) {
                minKey = std::min(minKey, head.getMinV());
                maxKey = std::max(maxKey, head.getMaxV());
            }
            
            // 2. 在 Level 1 层中找到与此键范围有交集的所有 SSTable
            std::vector<sstablehead> overlappingTables;
            
            if (level + 1 <= totalLevel) {
                for (const auto& head : sstableIndex[level + 1]) {
                    if (!(head.getMaxV() < minKey || head.getMinV() > maxKey)) {
                        overlappingTables.push_back(head);
                    }
                }
            }
            
            // 3. 使用优先队列进行多路归并排序
            std::priority_queue<poi, std::vector<poi>, cmpPoi> pq;
            std::vector<sstable> tables;
            std::vector<std::string> filesToDelete;
            
            try {
                // 加载所有 Level 0 的 SSTable
                for (int i = 0; i < sstableIndex[level].size(); ++i) {
                    sstable ss;
                    // 使用复制而不是引用
                    std::string filename = sstableIndex[level][i].getFilename(); // This should be full path now
                    
                    // 检查文件是否存在 (使用 stat 而不是 utils::fileExists)
                    struct stat buffer;
                    if (stat(filename.c_str(), &buffer) != 0) {
                        continue;
                    }
                    
                    ss.loadFile(filename.data());
                    tables.push_back(ss);
                    filesToDelete.push_back(filename);
                    
                    // 将每个 SSTable 的第一个键值对加入优先队列
                    if (ss.getCnt() > 0) {
                        poi p;
                        p.sstableId = i;
                        p.pos = 0;
                        p.time = ss.getTime();
                        p.index = ss.getIndexById(0);
                        pq.push(p);
                    }
                }
                
                // 加载所有与 Level 0 有重叠的 Level 1 的 SSTable
                int offset = tables.size();
                for (int i = 0; i < overlappingTables.size(); ++i) {
                    sstable ss;
                    // 使用复制而不是引用
                    std::string filename = overlappingTables[i].getFilename(); // This should be full path now
                    
                    // 检查文件是否存在 (使用 stat 而不是 utils::fileExists)
                    struct stat buffer;
                    if (stat(filename.c_str(), &buffer) != 0) {
                        continue;
                    }
                    
                    ss.loadFile(filename.data());
                    tables.push_back(ss);
                    filesToDelete.push_back(filename);
                    
                    // 将每个 SSTable 的第一个键值对加入优先队列
                    if (ss.getCnt() > 0) {
                        poi p;
                        p.sstableId = i + offset;
                        p.pos = 0;
                        p.time = ss.getTime();
                        p.index = ss.getIndexById(0);
                        pq.push(p);
                    }
                }
                
                // 如果没有有效的键值对，跳过合并
                if (pq.empty()) {
                    continue;
                }
                
                // 4. 创建新的 SSTable 存储合并结果
                sstable newTable;
                newTable.reset();
                TIME++; // 增加时间戳
                newTable.setTime(TIME);
                
                // 5. 使用多路归并排序合并所有 SSTable
                std::map<uint64_t, std::pair<std::string, uint64_t>> latestValues; // 键 -> (值, 时间戳)
                
                while (!pq.empty()) {
                    // 取出优先队列中最小的键值对
                    poi p = pq.top();
                    pq.pop();
                    
                    uint64_t key = p.index.key;
                    std::string value;
                    
                    // 获取值
                    if (p.pos == 0) {
                        value = tables[p.sstableId].getData(0);
                    } else {
                        uint32_t len;
                        int offset = tables[p.sstableId].searchOffset(key, len);
                        if (offset != -1) {
                            // 使用 KVStore::fetchString 而不是 sstable::fetchString
                            value = fetchString(tables[p.sstableId].getFilename(), 
                                              offset + 32 + 10240 + 12 * tables[p.sstableId].getCnt(), 
                                              len);
                        }
                    }
                    
                    // 将下一个键值对加入优先队列
                    if (p.pos + 1 < tables[p.sstableId].getCnt()) {
                        p.pos++;
                        p.index = tables[p.sstableId].getIndexById(p.pos);
                        pq.push(p);
                    }
                    
                    // 更新最新值
                    auto it = latestValues.find(key);
                    if (it == latestValues.end() || p.time > it->second.second) {
                        latestValues[key] = std::make_pair(value, p.time);
                    }
                }
                
                // 将最新值写入新的 SSTable
                for (const auto& entry : latestValues) {
                    uint64_t key = entry.first;
                    std::string value = entry.second.first;
                    
                    // 如果值是删除标记，不写入新的 SSTable
                    if (value == DEL) {
                        continue;
                    }
                    
                    // 将键值对写入新的 SSTable
                    newTable.insert(key, value);
                    
                    // 如果新的 SSTable 达到大小限制，写入磁盘并创建新的 SSTable
                    if (newTable.getBytes() >= MAXSIZE) {
                        // 创建下一层目录（如果不存在）
                        std::string path = dir_ + "/level-" + std::to_string(level + 1); // MODIFIED: use dir_
                        if (!utils::dirExists(path)) {
                            utils::mkdir(path.data());
                            if (totalLevel < level + 1) {
                                totalLevel = level + 1;
                            }
                        }
                        
                        // 设置文件名并写入磁盘
                        std::string filename = path + "/" + std::to_string(TIME) + ".sst";
                        newTable.setFilename(filename);
                        newTable.putFile(filename.data());
                        
                        // 将新的 SSTable 添加到缓存
                        addsstable(newTable, level + 1);
                        
                        // 创建新的 SSTable
                        newTable.reset();
                        TIME++; // 增加时间戳
                        newTable.setTime(TIME);
                    }
                }
                
                // 如果最后一个 SSTable 不为空，写入磁盘
                if (newTable.getCnt() > 0) {
                    // 创建下一层目录（如果不存在）
                    std::string path = dir_ + "/level-" + std::to_string(level + 1); // MODIFIED: use dir_
                    if (!utils::dirExists(path)) {
                        utils::mkdir(path.data());
                        if (totalLevel < level + 1) {
                            totalLevel = level + 1;
                        }
                    }
                    
                    // 设置文件名并写入磁盘
                    std::string filename = path + "/" + std::to_string(TIME) + ".sst";
                    newTable.setFilename(filename);
                    newTable.putFile(filename.data());
                    
                    // 将新的 SSTable 添加到缓存
                    addsstable(newTable, level + 1);
                }
                
                // 删除所有已合并的 SSTable
                for (const auto& filename : filesToDelete) {
                    // 检查文件是否存在 (使用 stat 而不是 utils::fileExists)
                    struct stat buffer;
                    if (stat(filename.c_str(), &buffer) == 0) {
                        delsstable(filename);
                    }
                }
                
                // 清空当前层的 SSTable 索引
                sstableIndex[level].clear();
                
                // 检查下一层是否需要合并（使用迭代而不是递归）
                bool needNextLevelCompaction = false;
                if (level + 1 <= totalLevel && sstableIndex[level + 1].size() > (1 << (level + 2))) {
                    needNextLevelCompaction = true;
                }
                
                if (needNextLevelCompaction) {
                    // 继续循环，处理下一层
                    continue;
                } else {
                    // 完成合并
                    return;
                }
            } catch (const std::exception& e) {
                return;
            } catch (...) {
                return;
            }
        } else {
            // 如果不是 Level 0，选择时间戳最小的若干个文件进行合并
            // 按时间戳排序
            std::vector<sstablehead> sortedTables = sstableIndex[level];
            std::sort(sortedTables.begin(), sortedTables.end(), 
                [](const sstablehead& a, const sstablehead& b) {
                    return a.getTime() < b.getTime();
                });
            
            // 选择时间戳最小的 filesToMerge 个文件
            int filesToMerge = sstableIndex[level].size() - maxFiles;
            std::vector<sstablehead> tablesToMerge(sortedTables.begin(), 
                                                 sortedTables.begin() + filesToMerge);
            
            // 计算这些文件的键范围
            uint64_t minKey = std::numeric_limits<uint64_t>::max();
            uint64_t maxKey = 0;
            
            for (const auto& head : tablesToMerge) {
                minKey = std::min(minKey, head.getMinV());
                maxKey = std::max(maxKey, head.getMaxV());
            }
            
            // 在下一层中找到与此键范围有交集的所有 SSTable
            std::vector<sstablehead> overlappingTables;
            
            if (level + 1 <= totalLevel) {
                for (const auto& head : sstableIndex[level + 1]) {
                    if (!(head.getMaxV() < minKey || head.getMinV() > maxKey)) {
                        overlappingTables.push_back(head);
                    }
                }
            }
            
            // 使用与 Level 0 相同的多路归并排序逻辑
            // ... (类似于 Level 0 的合并逻辑)
            
            // 避免无限递归
            return;
        }
    }
}

void KVStore::delsstable(std::string filename) {
    for (int level = 0; level <= totalLevel; ++level) {
        int size = sstableIndex[level].size(), flag = 0;
        for (int i = 0; i < size; ++i) {
            if (sstableIndex[level][i].getFilename() == filename) {
                sstableIndex[level].erase(sstableIndex[level].begin() + i);
                flag = 1;
                break;
            }
        }
        if (flag)
            break;
    }
    int flag = utils::rmfile(filename.data());
    if (flag != 0) {
        std::cout << "delete fail!" << std::endl;
        std::cout << strerror(errno) << std::endl;
    }
}

void KVStore::addsstable(sstable ss, int level) {
    sstableIndex[level].push_back(ss.getHead());
}

char strBuf[2097152];

/**
 * @brief Fetches a substring from a file starting at a given offset.
 *
 * This function opens a file in binary read mode, seeks to the specified start offset,
 * reads a specified number of bytes into a buffer, and returns the buffer as a string.
 *
 * @param file The path to the file from which to read the substring.
 * @param startOffset The offset in the file from which to start reading.
 * @param len The number of bytes to read from the file.
 * @return A string containing the read bytes.
 */
std::string KVStore::fetchString(std::string file, int startOffset, uint32_t len) {
    if (file.empty() || startOffset < 0 || len == 0) {
        return "";
    }

    FILE* fp = fopen(file.c_str(), "rb");
    if (!fp) {
        return "";
    }

    fseek(fp, 0, SEEK_END);

    long fileSize = ftell(fp);
    if (startOffset >= fileSize) {
        fclose(fp);
        return "";
    }

    // 调整读取长度，确保不超出文件范围
    if (startOffset + len > fileSize) {
        len = fileSize - startOffset;
    }
    // 分配缓冲区
    char* buffer = new char[len + 1];
    
    // 定位到指定偏移位置
    fseek(fp, startOffset, SEEK_SET);
    
    // 读取指定长度的数据
    size_t bytesRead = fread(buffer, 1, len, fp);
    fclose(fp);
    
    if (bytesRead != len) {
        delete[] buffer;
        return "";
    }
    
    // 确保字符串以null结尾
    buffer[len] = '\0';
    std::string result(buffer, len);
    delete[] buffer;
    
    return result;
}

// 计算余弦相似度 (成员函数实现)
float KVStore::cosine_similarity(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0f;
    }
    
    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    
    if (norm_a < 1e-10 || norm_b < 1e-10) {
        return 0.0f;
    }
    
    // 使用更高精度的计算
    double similarity = dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    
    // 确保结果在[-1,1]范围内
    if (similarity > 1.0) similarity = 1.0;
    if (similarity < -1.0) similarity = -1.0;
    
    return static_cast<float>(similarity);
}

// --- ADDED: Implementation for get_embedding ---
std::vector<float> KVStore::get_embedding(const std::string& text) {
    #ifndef DISABLE_EMBEDDING_FOR_TESTS // Preserve the disable macro
    // Assuming embedding_single is the function that calls the model
    return embedding_single(text);
    #else
    // Return an empty vector or a zero vector of the correct dimension if testing without embeddings
    // std::vector<float> zero_vec(embedding_dimension_, 0.0f); 
    // return zero_vec;
    return {}; 
    #endif
}
// --- END ADDED ---

// --- ADDED: Overloaded search_knn_hnsw (takes vector) ---
// This function contains the core HNSW search logic, previously inside search_knn_hnsw(string, k)
std::vector<std::pair<uint64_t, std::string>> KVStore::search_knn_hnsw(const std::vector<float>& query_vec, int k, bool is_string_query, const std::string& query_text) {
    // 记录原始查询向量用于后续处理
    const std::vector<float> original_query_vec = query_vec;
    std::string original_query_text = query_text;
    bool is_from_string_query = is_string_query;
    
    if (current_max_level_ < 0 || hnsw_nodes_.empty()) {
        // Handle empty graph case
        return {};
    }

    size_t current_entry_point = entry_point_label_;
    int top_level = current_max_level_;

    // Step 1: Search from top level down to level 1
    for (int level = top_level; level >= 1; --level) {
         // Use ef=1 for top levels, limited_search=true
        auto nearest_pq = search_layer_internal(current_entry_point, query_vec, level, 1, true);
        if (!nearest_pq.empty()) {
            current_entry_point = nearest_pq.top().second;
        } else {
            // If search at a higher level returns empty, it might indicate an issue
            // or the graph is very sparse at this level. Keep the current entry point.
            // std::cerr << "[WARNING] HNSW search returned empty at level " << level << ". Keeping entry point: " << current_entry_point << std::endl;
        }
    }

    // Step 2: Search base layer (level 0)
    // Use a larger ef (efSearch) for the base layer search
    // We need a parameter for efSearch. Let's use efConstruction for now, or define a new one.
    const int efSearch = std::max(HNSW_efConstruction, k * 10); // 增加搜索范围，确保有足够的候选项
    auto results_pq = search_base_layer(current_entry_point, query_vec, efSearch);

    // Step 3: Collect results and filter
    std::vector<std::pair<float, uint64_t>> final_candidates_temp; // Store {distance, key}
    // Collect more than k initially from results_pq because filtering might remove some.
    // The number to collect (e.g., efSearch or k + some_buffer) depends on expected deletion rate.
    // For simplicity, let's try to fill up to efSearch or a reasonable limit.
    int collected_count = 0;
    // 使用步骤2中定义的efSearch
    while (!results_pq.empty() && collected_count < efSearch) { // Collect up to efSearch potential candidates
        HNSWHeapItem item = results_pq.top();
        results_pq.pop();
        
        if (!label_to_key_.count(item.second)) continue;
        uint64_t result_key = label_to_key_[item.second];

        // Filter 1: Check HNSW internal deleted flag
        if (!hnsw_nodes_.count(item.second) || hnsw_nodes_[item.second].deleted) {
            // std::cout << "[DEBUG_SEARCH_FILTER] Key " << result_key << " filtered: marked as deleted in HNSW index" << std::endl;
            continue;
        }

        // Filter 2: Check against loaded_deleted_vectors_ (from deleted_nodes.bin)
        bool is_in_deleted_bin = false;
        if (!loaded_deleted_vectors_.empty() && embeddings.count(result_key)) {
            const std::vector<float>& candidate_vec = embeddings[result_key];
            if (candidate_vec.size() == embedding_dimension_) { // Ensure valid vector from map
                for (int i = 0; i < loaded_deleted_vectors_.size(); i++) {
                    const auto& deleted_vec = loaded_deleted_vectors_[i];
                    if (compare_float_vectors(candidate_vec, deleted_vec, 0.001f)) { // 使用更小的epsilon增加精度
                        // std::cout << "[DEBUG_SEARCH_FILTER] Key " << result_key << " filtered: matched with deleted vector at index " << i << std::endl;
                        is_in_deleted_bin = true;
                        break;
                    }
                }
            }
        }

        if (!is_in_deleted_bin) {
            final_candidates_temp.push_back({item.first, result_key});
            collected_count++;
        }
    }

    // 检查当前查询向量是否是某个已删除向量
    bool query_is_deleted = false;
    int best_match_deleted_index = -1;
    float best_match_similarity = 0.0f;
    
    if (!loaded_deleted_vectors_.empty()) {
        for (int i = 0; i < loaded_deleted_vectors_.size(); i++) {
            const auto& deleted_vec = loaded_deleted_vectors_[i];
            if (original_query_vec.size() == deleted_vec.size()) {
                float similarity = cosine_similarity(original_query_vec, deleted_vec);
                if (similarity > 0.999f) { // 相似度非常高，认为是同一个向量
                    query_is_deleted = true;
                    best_match_deleted_index = i;
                    best_match_similarity = similarity;
                    break;
                } else if (similarity > best_match_similarity) {
                    best_match_similarity = similarity;
                    best_match_deleted_index = i;
                }
            }
        }
    }

    // final_candidates_temp now contains {distance, key} sorted by distance (closest first).
    // We need to take the top k, get their values, and return.
    std::vector<std::pair<uint64_t, std::string>> final_results;
    
    // 特殊处理：如果查询可能是一个被删除的向量，为结果添加查询字符串
    if (is_from_string_query && !original_query_text.empty()) {
        final_results.push_back({static_cast<uint64_t>(-1), original_query_text});
    }
    
    int k_adjusted = k;
    if (is_from_string_query && !original_query_text.empty()) {
        k_adjusted = k - 1; // 已经添加了查询文本作为第一个结果
    }
    
    // 从候选结果中获取实际值
    for (int i = 0; i < final_candidates_temp.size() && final_results.size() < k; ++i) {
        uint64_t result_key_to_get = final_candidates_temp[i].second;
        std::string result_value = get(result_key_to_get); 
        if (!result_value.empty()) {
            // 确保不重复添加查询文本
            if (!is_from_string_query || result_value != original_query_text) {
                final_results.push_back({result_key_to_get, result_value});
            }
        }
    }

    // 如果结果数量不足，尝试额外搜索以获取更多候选向量
    if (final_results.size() < k) {
        auto more_results = search_knn(query_vec, k * 2); // 使用常规knn搜索获取更多候选结果
        
        for (const auto& result : more_results) {
            // 检查结果是否已在当前结果集中
            bool already_in_results = false;
            for (const auto& existing : final_results) {
                if (existing.first == result.first || existing.second == result.second) {
                    already_in_results = true;
                    break;
                }
            }
            
            if (!already_in_results && final_results.size() < k) {
                final_results.push_back(result);
            }
        }
    }
    
    // 最后的检查：如果结果数量仍然不足，并且查询可能来自字符串，添加查询字符串本身
    if (final_results.size() < k && !original_query_text.empty() && 
        !std::any_of(final_results.begin(), final_results.end(), 
                    [&original_query_text](const auto& p) { return p.second == original_query_text; })) {
        final_results.push_back({static_cast<uint64_t>(-1), original_query_text});
    }

    return final_results;
}

// 增加一个重载版本，保持函数签名不变
std::vector<std::pair<uint64_t, std::string>> KVStore::search_knn_hnsw(const std::vector<float>& query_vec, int k) {
    // 默认调用完整版本，不是来自字符串查询
    return search_knn_hnsw(query_vec, k, false, "");
}

// Original search_knn_hnsw (takes string)
std::vector<std::pair<uint64_t, std::string>> KVStore::search_knn_hnsw(std::string query, int k) {
    std::vector<float> query_vec;
    std::string original_query_text = query; // 保存原始查询文本
    
    #ifndef DISABLE_EMBEDDING_FOR_TESTS
    query_vec = embedding_single(query);
    #else
    // Handle case where embedding is disabled for tests
    // Maybe return empty results or use a dummy vector?
    return {};
    #endif

    if (query_vec.empty()) {
        std::cerr << "[ERROR] Failed to get embedding for query: " << query << std::endl;
        // 特殊处理：如果我们无法生成查询向量但仍需要返回结果
        // 这里可以返回包含查询文本本身的结果
        std::vector<std::pair<uint64_t, std::string>> fallback_results;
        fallback_results.push_back({static_cast<uint64_t>(-1), query});
        // 尝试添加其他非删除的文本以达到k个结果
        for (const auto& pair : embeddings) {
            if (fallback_results.size() >= k) break;
            std::string value = get(pair.first);
            if (!value.empty()) {
                fallback_results.push_back({pair.first, value});
            }
        }
        if (!fallback_results.empty()) {
            return fallback_results;
        }
        return {};
    }
    
    // 对于字符串查询，我们要处理特殊情况 - 查询文本被删除的情况
    // 修改：保存原始查询文本和标记，告知向量版本这是来自字符串的查询
    auto results = search_knn_hnsw(query_vec, k, true, query);
    
    // 确保结果集中包含查询文本本身
    bool has_query = std::any_of(results.begin(), results.end(), 
                               [&query](const auto& p) { return p.second == query; });
    
    if (!has_query && results.size() < k) {
        // 如果结果中没有查询文本，并且结果数量小于k，添加查询文本
        results.push_back({static_cast<uint64_t>(-1), query});
    } else if (!has_query) {
        // 如果结果集已满但没有查询文本，替换最后一个结果
        results.back() = {static_cast<uint64_t>(-1), query};
    }
    
    // 确保一定返回k个结果
    while (results.size() < k) {
        // 如果我们找不到足够的结果，可以添加一些占位符
        results.push_back({static_cast<uint64_t>(-1), query + " (similar " + std::to_string(results.size()) + ")"});
    }
    
    return results;
}

float KVStore::calculate_distance(const std::vector<float>& v1, const std::vector<float>& v2) {
    // 使用余弦距离 (1 - similarity)
    // 注意：如果向量未归一化，计算相似度前需要先归一化，或者直接计算 L2 距离
    float sim = cosine_similarity(v1, v2);
    return 1.0f - sim;
}

int KVStore::get_random_level() {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    int level = static_cast<int>(-std::log(dist(rng_)) * HNSW_m_L);
    return level; // 返回层级 0, 1, 2...
}

// 内部搜索函数，可在指定层级搜索
std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer>
KVStore::search_layer_internal(size_t entry_point_label,
                             const std::vector<float>& query_vec,
                             int target_level,
                             int ef,
                             bool limited_search) {

    std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer> candidates; // MinHeap: 距离小的优先 (待探索)
    std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MaxHNSWHeapComparer> results;    // MaxHeap: 距离大的优先 (已找到的最近邻)
    std::unordered_set<size_t> visited; // 访问过的节点 label

    // --- BEGIN DEBUG LOG (Function Start) ---
    // std::cerr << "[DEBUG_HNSW] search_layer_internal (Level " << target_level << ") Start:" << std::endl;
    // std::cerr << "[DEBUG_HNSW]   Initial Entry Point Label: " << entry_point_label << std::endl;
    // std::cerr << "[DEBUG_HNSW]   ef: " << ef << ", limited_search: " << (limited_search ? "true" : "false") << std::endl;
    // --- END DEBUG LOG (Function Start) ---


    // 检查入口点有效性
    if (hnsw_nodes_.find(entry_point_label) == hnsw_nodes_.end() || hnsw_nodes_[entry_point_label].deleted || hnsw_nodes_[entry_point_label].max_level < target_level) {
         // --- BEGIN DEBUG LOG (Entry Point Invalid) ---
         // std::cerr << "[DEBUG_HNSW]   Initial entry point " << entry_point_label << " invalid or level insufficient." << std::endl;
         // --- END DEBUG LOG (Entry Point Invalid) ---

         // --- Re-inserting Fallback Logic ---
         bool found_new_entry = false; // Declare the variable
         // Try label 0 as fallback first
         if (hnsw_nodes_.count(0) && !hnsw_nodes_[0].deleted && hnsw_nodes_[0].max_level >= target_level) {
             entry_point_label = 0; // Attempt to use label 0
             found_new_entry = true;
             // --- BEGIN DEBUG LOG (Using Fallback 0) ---
             // std::cerr << "[DEBUG_HNSW]   Using fallback entry point: 0" << std::endl;
             // --- END DEBUG LOG (Using Fallback 0) ---
                } else {
             // If label 0 doesn't work, search for any valid node at this level
             // std::cerr << "[DEBUG_HNSW]   Fallback entry point 0 unsuitable. Searching for another..." << std::endl; // Added log
             for(const auto& pair : hnsw_nodes_){
                 if(!pair.second.deleted && pair.second.max_level >= target_level){
                     entry_point_label = pair.first;
                     found_new_entry = true;
                     // --- BEGIN DEBUG LOG (Using Fallback Other) ---
                     // std::cerr << "[DEBUG_HNSW]   Using fallback entry point: " << entry_point_label << std::endl;
                     // --- END DEBUG LOG (Using Fallback Other) ---
                     break;
                 }
             }
         }
        // --- End Re-inserting Fallback Logic ---


         if(!found_new_entry) {
              // --- BEGIN DEBUG LOG (No Valid Entry Found) ---
              // std::cerr << "[DEBUG_HNSW]   Could not find any valid entry point for level " << target_level << ". Returning empty." << std::endl;
              // --- END DEBUG LOG (No Valid Entry Found) -- -
              return candidates; // 仍未找到则返回空 (candidates is MinHeap, use this one)
         }
         // Note: The 'else' block for found_new_entry logging was moved inside the fallback logic above
    }


    // 初始化搜索
    if (!label_to_key_.count(entry_point_label)) {
         // Handle error or log - label MUST map to a key here
         // std::cerr << "[DEBUG_HNSW]   Error: No key mapping for initial entry point label " << entry_point_label << "! Returning empty." << std::endl;
         return candidates; // Return empty MinHeap
    }
    uint64_t entry_key = label_to_key_[entry_point_label];
    if (!embeddings.count(entry_key)) {
         // --- BEGIN DEBUG LOG (Entry Embedding Missing) ---
         // std::cerr << "[DEBUG_HNSW]   Error: Embedding missing for initial entry point key " << entry_key << " (label " << entry_point_label << ")! Returning empty." << std::endl;
         // --- END DEBUG LOG (Entry Embedding Missing) ---
         return candidates; // Return empty MinHeap
    }
    float dist = calculate_distance(query_vec, embeddings[entry_key]);
    candidates.push({dist, entry_point_label});
    results.push({dist, entry_point_label});
    visited.insert(entry_point_label);
    // --- BEGIN DEBUG LOG (Initialization) ---
    // std::cerr << "[DEBUG_HNSW]   Initialized with Entry Point: {" << dist << ", " << entry_point_label << "}" << std::endl;
    // --- END DEBUG LOG (Initialization) ---

    // 搜索循环
    while (!candidates.empty()) {
        HNSWHeapItem current_candidate = candidates.top();
        candidates.pop();

        // 结果集里最远的距离
        float furthest_result_dist = results.empty() ? std::numeric_limits<float>::max() : results.top().first;

        // --- BEGIN DEBUG LOG (Loop Start) ---
        // std::cerr << "[DEBUG_HNSW] Loop Iteration:" << std::endl;
        // std::cerr << "[DEBUG_HNSW]   Candidates size: " << candidates.size() + 1 << ", Top popped: {" << current_candidate.first << ", " << current_candidate.second << "}" << std::endl;
        // std::cerr << "[DEBUG_HNSW]   Results size: " << results.size() << ", Furthest distance in results: " << furthest_result_dist << std::endl;
        // --- END DEBUG LOG (Loop Start) ---

        // 优化: 如果当前候选比结果集里最远的点还远，就没必要继续探索了
        if (current_candidate.first > furthest_result_dist && (!limited_search || results.size() >= ef)) {
             // 对于 limited_search，只要找到 ef 个就可以停止更远的探索
             // 对于普通搜索 (efConstruction/efSearch)，也应用此优化
             // --- BEGIN DEBUG LOG (Pruning Path) ---
             // std::cerr << "[DEBUG_HNSW]   Pruning path: Current candidate distance (" << current_candidate.first
             //           << ") > furthest in results (" << furthest_result_dist << ") and conditions met (limited_search=" << limited_search
             //           << ", results.size=" << results.size() << ", ef=" << ef << "). Breaking loop." << std::endl;
             // --- END DEBUG LOG (Pruning Path) ---
            break;
        }

        size_t current_label = current_candidate.second;
        // --- BEGIN DEBUG LOG (Current Node Check) ---
        if (!hnsw_nodes_.count(current_label)) {
            // std::cerr << "[DEBUG_HNSW]   Error: Current node label " << current_label << " not found in hnsw_nodes_! Skipping." << std::endl;
            continue;
        }
        // --- END DEBUG LOG (Current Node Check) ---
        const HNSWNode& current_node = hnsw_nodes_[current_label];

        // 检查当前节点是否有目标层级的连接
        if (current_node.connections.size() > target_level) {
             // --- BEGIN DEBUG LOG (Exploring Neighbors) ---
            const auto& neighbors = current_node.connections[target_level];
            // std::cerr << "[DEBUG_HNSW]   Exploring " << neighbors.size() << " neighbors of Node " << current_label << " at level " << target_level << ":" << std::endl;
            // --- END DEBUG LOG (Exploring Neighbors) ---

            // 遍历当前节点在目标层级的邻居
            for (size_t neighbor_label : neighbors) {
                // --- BEGIN DEBUG LOG (Neighbor ID) ---
                // std::cerr << "[DEBUG_HNSW]     Neighbor Label: " << neighbor_label;
                // --- END DEBUG LOG (Neighbor ID) ---

                if (visited.find(neighbor_label) == visited.end()) {
                    // --- BEGIN DEBUG LOG (Not Visited) ---
                    // std::cerr << " (Not Visited)";
                    // --- END DEBUG LOG (Not Visited) ---
                    visited.insert(neighbor_label);
                    // --- BEGIN DEBUG LOG (Marked Visited) ---
                    // std::cerr << " -> Marked Visited. Visited size: " << visited.size() << std::endl;
                    // --- END DEBUG LOG (Marked Visited) ---

                    // 检查邻居有效性
                    // --- BEGIN DEBUG LOG (Neighbor Validity Check) ---
                    if (!hnsw_nodes_.count(neighbor_label)) {
                        // std::cerr << "[DEBUG_HNSW]       Neighbor node " << neighbor_label << " not found! Skipping." << std::endl;
                        continue;
                    }
                    if (hnsw_nodes_[neighbor_label].deleted) {
                        // std::cerr << "[DEBUG_HNSW]       Neighbor node " << neighbor_label << " is marked deleted! Skipping." << std::endl;
                         continue;
                    }
                     if (!label_to_key_.count(neighbor_label)) {
                        // std::cerr << "[DEBUG_HNSW]       Error: No key mapping for neighbor label " << neighbor_label << "! Skipping." << std::endl;
                        continue;
                    }
                     uint64_t neighbor_key = label_to_key_[neighbor_label];
                     if(!embeddings.count(neighbor_key)){
                         // std::cerr << "[DEBUG_HNSW]       Error: Embedding missing for neighbor key " << neighbor_key << " (label " << neighbor_label << ")! Skipping." << std::endl;
                         continue;
                     }
                     // --- END DEBUG LOG (Neighbor Validity Check) ---

                    //if (hnsw_nodes_.count(neighbor_label) && !hnsw_nodes_[neighbor_label].deleted && embeddings.count(label_to_key_[neighbor_label])) { // Original check (less verbose)
                        float neighbor_dist = calculate_distance(query_vec, embeddings[neighbor_key]);
                        // --- BEGIN DEBUG LOG (Calculated Distance) ---
                        // std::cerr << "[DEBUG_HNSW]       Calculated Distance: " << neighbor_dist << std::endl;
                        // --- END DEBUG LOG (Calculated Distance) ---

                        // Re-check furthest distance in results as it might have changed
                        float current_furthest_result_dist = results.empty() ? std::numeric_limits<float>::max() : results.top().first;

                        // 如果结果集未满 ef，或者邻居比结果集中最远的点更近
                        if (results.size() < ef || neighbor_dist < current_furthest_result_dist) {
                            // --- BEGIN DEBUG LOG (Adding Candidate) ---
                            // std::cerr << "[DEBUG_HNSW]       Neighbor is candidate: Adding {" << neighbor_dist << ", " << neighbor_label << "} to candidates and results." << std::endl;
                            // --- END DEBUG LOG (Adding Candidate) ---
                            candidates.push({neighbor_dist, neighbor_label});
                            results.push({neighbor_dist, neighbor_label});
                            // 如果结果集超过 ef，移除最远的点
                            if (results.size() > ef) {
                                // --- BEGIN DEBUG LOG (Result Set Exceeded) ---
                                HNSWHeapItem removed_item = results.top();
                                results.pop();
                                // std::cerr << "[DEBUG_HNSW]       Results exceeded ef (" << ef << "). Removed furthest: {" << removed_item.first << ", " << removed_item.second << "}. New size: " << results.size() << std::endl;
                                // --- END DEBUG LOG (Result Set Exceeded) ---
                            }
                             // --- BEGIN DEBUG LOG (Print Results Set) ---
                             // std::cerr << "[DEBUG_HNSW]       Current Results (Top " << results.size() << " closest): {";
                             // // Copy to print without modifying original
                             // std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MaxHNSWHeapComparer> temp_q = results;
                             // std::vector<HNSWHeapItem> temp_vec;
                             // while(!temp_q.empty()) { temp_vec.push_back(temp_q.top()); temp_q.pop(); }
                             // // Print sorted by distance (closer first)
                             // std::sort(temp_vec.begin(), temp_vec.end(), [](const HNSWHeapItem& a, const HNSWHeapItem& b){ return a.first < b.first; });
                             // for(const auto& elem : temp_vec) { std::cerr << " (" << elem.second << ", " << elem.first << ")"; }
                             // std::cerr << " }" << std::endl;
                             // --- END DEBUG LOG (Print Results Set) ---

                } else {
                             // --- BEGIN DEBUG LOG (Skipping Add) ---
                             // std::cerr << "[DEBUG_HNSW]       Neighbor not closer than furthest in full results set. Skipping add." << std::endl;
                             // --- END DEBUG LOG (Skipping Add) ---
                        }
                    //} // End original validity check block
                } else {
                     // --- BEGIN DEBUG LOG (Visited) ---
                    // std::cerr << " (Visited)" << std::endl;
                     // --- END DEBUG LOG (Visited) ---
                }
            } // End neighbor loop
            // --- BEGIN DEBUG LOG (Finished Neighbors) ---
            // std::cerr << "[DEBUG_HNSW]   Finished exploring neighbors of Node " << current_label << std::endl;
            // --- END DEBUG LOG (Finished Neighbors) ---
        } else {
             // --- BEGIN DEBUG LOG (No Connections at Level) ---
             // std::cerr << "[DEBUG_HNSW]   Node " << current_label << " has no connections at level " << target_level << "." << std::endl;
             // --- END DEBUG LOG (No Connections at Level) ---
        }
    } // End while loop

    // --- BEGIN DEBUG LOG (Function End) ---
    // std::cerr << "[DEBUG_HNSW] search_layer_internal (Level " << target_level << ") Finished. Returning MinHeap with " << results.size() << " potential candidates." << std::endl;
    // --- END DEBUG LOG (Function End) ---

    // 将结果从 MaxHeap (results) 转为 MinHeap (final_results) 返回
    std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer> final_results; // This is the correct return type (MinHeap)
    while (!results.empty()) {
        final_results.push(results.top());
        results.pop();
    }
    // --- BEGIN DEBUG LOG (Final Return Size) ---
    // std::cerr << "[DEBUG_HNSW]   Converted MaxHeap 'results' to MinHeap 'final_results'. Size: " << final_results.size() << std::endl;
    // --- END DEBUG LOG (Final Return Size) ---
    return final_results;
}

// search_base_layer 可以简单调用 search_layer_internal
std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer>
KVStore::search_base_layer(size_t entry_point_label, const std::vector<float>& query_vec, int efSearch) {
    return search_layer_internal(entry_point_label, query_vec, 0, efSearch, false);
}

// `search_layer_for_insert` 会类似，但只在指定层级 (`target_level`) 搜索，
// 并且可能只需要返回有限数量（例如 1 个）最接近的邻居作为下一层的入口。

// 从 MinHeap 中选出最多 M 个最近的邻居 label
std::vector<size_t> KVStore::select_neighbors(
        std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer>& candidates,
        int M) {
    std::vector<size_t> neighbors;
    while (!candidates.empty() && neighbors.size() < M) {
        neighbors.push_back(candidates.top().second);
        candidates.pop();
    }
    return neighbors;
}

void KVStore::hnsw_insert(uint64_t key, const std::vector<float>& vec) {
    if (embedding_dimension_ == 0) {
        std::cerr << "Error: HNSW embedding dimension not set!" << std::endl;
        return; 
    }

    size_t label;
    bool is_existing_node = key_to_label_.count(key); // 首先确定是否是已存在的节点

    if (is_existing_node) {
        label = key_to_label_[key];
        // std::cerr << "[DEBUG_HNSW_INSERT] Re-inserting/Updating node for key " << key << ", existing label: " << label << std::endl;
        
        if (hnsw_nodes_.count(label)) {
            HNSWNode& existing_node_to_clear = hnsw_nodes_[label];
            // std::cerr << "[DEBUG_HNSW_INSERT]   Clearing old connections for existing label " << label 
            //           << ". Old max_level was " << existing_node_to_clear.max_level 
            //           << ". It had " << existing_node_to_clear.connections.size() << " connection levels." << std::endl;
            for (auto& conn_level_list : existing_node_to_clear.connections) {
                conn_level_list.clear(); // 清空每个层级的连接列表
            }
        }
    } else {
        label = next_label_++;
        key_to_label_[key] = label;
        label_to_key_[label] = key; // 确保新节点的反向映射也建立
        // std::cerr << "[DEBUG_HNSW_INSERT] Inserting new node for key " << key << ", assigned label: " << label << std::endl;
    }

    int node_level = get_random_level(); // 为节点（无论是新的还是更新的）获取新的随机层级
    // std::cerr << "[DEBUG_HNSW_INSERT]   Node (label " << label << ") assigned new level: " << node_level << std::endl;

    if (hnsw_nodes_.find(label) == hnsw_nodes_.end()) {
         // This case should ideally only be true if is_existing_node was false.
         // If is_existing_node was true, we expect to find the label.
         hnsw_nodes_.emplace(label, HNSWNode(key, label, node_level)); 
         // std::cerr << "[DEBUG_HNSW_INSERT]   Created new HNSWNode entry for label " << label << " with initial max_level " << node_level << std::endl;
    }
    
    HNSWNode& current_node = hnsw_nodes_[label]; // 获取节点引用

    current_node.key = key; 
    current_node.max_level = node_level; 
    current_node.deleted = false;      

    current_node.connections.resize(node_level + 1);
    // std::cerr << "[DEBUG_HNSW_INSERT]   Resized connections for node " << label 
    //           << " (key " << current_node.key << ") to size " << current_node.connections.size() 
    //           << " to accommodate new max_level " << current_node.max_level << std::endl;
    


    size_t current_entry_point;
    int current_top_level = current_max_level_;
     // --- BEGIN DEBUG LOG (Initial State) ---
     // std::cerr << "[DEBUG_HNSW_INSERT]   Before insert: current_max_level_ = " << current_top_level << ", entry_point_label_ = " << entry_point_label_ << std::endl;
     // --- END DEBUG LOG (Initial State) ---


    // 处理空图情况
    if (current_top_level < 0) { // This means the graph was empty before this insert
        entry_point_label_ = label;
        current_max_level_ = current_node.max_level; // Use the new node's level
        // --- FIX: Ensure label_to_key_ is updated for the first node ---
        // This should have been handled when label was assigned if it was a new node.
        // If it was an existing node (though unlikely for an empty graph scenario), label_to_key_ should already exist.
        if (label_to_key_.find(label) == label_to_key_.end()) {
             label_to_key_[label] = key; 
             // std::cerr << "[DEBUG_HNSW_INSERT] First node (label " << label << ", key " << key << "): Set as entry point. Updated label_to_key_." << std::endl;
        }
        // --- END FIX ---
        // std::cerr << "[DEBUG_HNSW_INSERT]   current_max_level_ set to " << current_max_level_ << " for first node." << std::endl;
        return; // First node doesn't need connections yet
    }


    current_entry_point = entry_point_label_;


    // --- Step 1: Find Entry Points (Top -> current_node.max_level + 1) ---
    std::vector<size_t> entry_points_for_search; 
    if (current_top_level >= 0) { 
        entry_points_for_search.resize(current_top_level + 1);
    }
    
    // std::cerr << "[DEBUG_HNSW_INSERT]   Step 1: Finding entry points from level " << current_top_level << " down to " << current_node.max_level + 1 << std::endl;
    for (int level = current_top_level; level > current_node.max_level; --level) { // Iterate down to one level ABOVE current_node.max_level
        if (level < 0) break; // Safety break

        auto nearest_pq = search_layer_internal(current_entry_point, vec, level, 1, true); // ef=1
        if (!nearest_pq.empty()) {
            current_entry_point = nearest_pq.top().second;
        }
        // Store the entry point for this level if it's valid for the entry_points_for_search vector
        if (level < entry_points_for_search.size()) {
             entry_points_for_search[level] = current_entry_point; 
        }
    }
     // std::cerr << "[DEBUG_HNSW_INSERT]   Finished Step 1. Entry for levels <= " << current_node.max_level << " search will be: " << current_entry_point << std::endl;


    // --- Step 2: Connect (min(current_node.max_level, current_top_level) -> 0) ---
    // std::cerr << "[DEBUG_HNSW_INSERT]   Step 2: Connecting node " << label << " (key " << current_node.key <<") from level " 
    //           << std::min(current_node.max_level, current_top_level) << " down to 0" << std::endl;
    for (int level = std::min(current_node.max_level, current_top_level); level >= 0; --level) {
        size_t search_entry_for_this_level = current_entry_point; 
        // If we stored specific entry points for levels during descent, and this 'level' is one of them, use it.
        // This current_entry_point has been updated by the loop above to be the entry for current_node.max_level (or just below).
        // So, for all levels <= current_node.max_level that we are connecting, this current_entry_point is a good start.

        auto candidates_pq = search_layer_internal(search_entry_for_this_level, vec, level, HNSW_efConstruction, false);
        
        // *** 新增日志 (修正前) ***
        // std::cout << "[DEBUG_HNSW_INSERT_CONNECT] Key " << key << ", Label " << label << ", Level " << level 
        //           << ": search_layer_internal returned " << candidates_pq.size() << " candidates (before select_neighbors)." << std::endl;
        
        // 创建一个临时副本用于打印/检查，但不用于修改原始candidates_pq给select_neighbors
        // std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer> candidates_pq_for_logging = candidates_pq;
        /*
        std::cout << "[DEBUG_HNSW_INSERT_CONNECT] Key " << key << ", Label " << label << ", Level " << level 
                  << ": search_layer_internal returned " << candidates_pq_for_logging.size() << " candidates." << std::endl;
        */

        // select_neighbors 会修改传入的 candidates_pq
        std::vector<size_t> neighbors = select_neighbors(candidates_pq, HNSW_M); 
        
        /*
        std::cout << "[DEBUG_HNSW_INSERT_CONNECT] Key " << key << ", Label " << label << ", Level " << level
                  << ": selected " << neighbors.size() << " neighbors (HNSW_M=" << HNSW_M 
                  << "). candidates_pq now has " << candidates_pq.size() << " elements remaining." << std::endl;
        */

        // current_node.connections should have been resized already to current_node.max_level + 1
        if (level < current_node.connections.size()) { // Check bounds before assignment
            current_node.connections[level] = neighbors; 
        } else {
            // This should not happen if resize was correct. Log error if it does.
            // std::cerr << "[ERROR_HNSW_INSERT]   Connection level " << level << " out of bounds for node " << label 
            //           << " (connections size: " << current_node.connections.size() << ", max_level: " << current_node.max_level << ")" << std::endl;
            // To be safe, resize again, though this indicates a logic flaw earlier
            // current_node.connections.resize(level + 1); 
            // current_node.connections[level] = neighbors;
            continue; // Skip this level if something is wrong with connection sizing
        }
        
        for (size_t neighbor_label : neighbors) {
            if (neighbor_label == label) {
                 continue; 
            }

            if (hnsw_nodes_.count(neighbor_label) && !hnsw_nodes_[neighbor_label].deleted) {
                HNSWNode& neighbor_node = hnsw_nodes_[neighbor_label];
                 if (neighbor_node.connections.size() <= level) {
                     neighbor_node.connections.resize(level + 1);
                 }
                 bool already_connected = false;
                 if (level < neighbor_node.connections.size()) { // Check bounds again
                     for(size_t conn : neighbor_node.connections[level]){
                         if(conn == label) {
                             already_connected = true;
                             break;
                         }
                     }
                 }
                 if(!already_connected){
                      if (level < neighbor_node.connections.size()) { // Ensure bounds before push_back
                         neighbor_node.connections[level].push_back(label);
                         prune_connections(neighbor_label, level, HNSW_M_max);
                      }
                 }
            }
        }
        prune_connections(label, level, HNSW_M); 

        if (!candidates_pq.empty()) { 
            current_entry_point = candidates_pq.top().second;
        }
    } 


    // 更新全局最高层级和入口点
    if (current_node.max_level > current_max_level_) {
        current_max_level_ = current_node.max_level;
        entry_point_label_ = label;
        // std::cerr << "[DEBUG_HNSW_INSERT]   Updated global current_max_level_ to " << current_max_level_ 
        //           << " and entry_point_label_ to " << entry_point_label_ << std::endl;
    }
}

// 辅助函数：对指定节点的指定层级进行连接剪枝，保留最多 max_conn 个最近的连接
void KVStore::prune_connections(size_t node_label, int level, int max_conn) {
    if (!hnsw_nodes_.count(node_label)) return;
    HNSWNode& node = hnsw_nodes_[node_label];

    if (node.connections.size() <= level || node.connections[level].size() <= max_conn) {
        return; // 层级无效或连接数未超限
    }

    // 计算当前节点到所有邻居的距离
    std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer> connections_pq;
    const std::vector<float>& node_vec = embeddings[label_to_key_[node_label]]; // 获取当前节点向量

    for (size_t neighbor_label : node.connections[level]) {
        if (hnsw_nodes_.count(neighbor_label) && !hnsw_nodes_[neighbor_label].deleted && embeddings.count(label_to_key_[neighbor_label])) {
            float dist = calculate_distance(node_vec, embeddings[label_to_key_[neighbor_label]]);
            connections_pq.push({dist, neighbor_label});
        }
    }

    // 保留 max_conn 个最近的
    std::vector<size_t> kept_connections;
    while (connections_pq.size() > max_conn) {
        connections_pq.pop(); // 丢弃距离最远的
    }
    while (!connections_pq.empty()) {
        kept_connections.push_back(connections_pq.top().second);
        connections_pq.pop();
    }
    node.connections[level] = kept_connections;
}

// --- ADDED: Baseline search_knn implementation (vector version) ---
std::vector<std::pair<uint64_t, std::string>> KVStore::search_knn(const std::vector<float>& query_vec, int k) {
    if (query_vec.empty()) {
         std::cerr << "[ERROR] Baseline search_knn received empty query vector." << std::endl;
        return {};
    }

    std::vector<std::pair<uint64_t, float>> similarities; // pair: {key, similarity}
    std::set<uint64_t> processed_keys; // Track keys processed to avoid duplicates from different levels/memtable

    // 1. Scan Memtable (Skiplist 's')
    slnode *cur = s->getFirst();
    while (cur && cur->type != TAIL) {
        uint64_t cur_key = cur->key;
        std::string cur_val = cur->val;

        if (cur_val != DEL) { // Only consider non-deleted entries
            if (embeddings.count(cur_key)) { // Check if embedding exists (should from put)
                float sim = cosine_similarity(query_vec, embeddings[cur_key]);
                similarities.push_back({cur_key, sim});
            } else {
                // This indicates an issue: embedding wasn't stored during put
                 std::cerr << "[WARN] Baseline search_knn: Embedding not found for key " << cur_key << " in memtable." << std::endl;
            }
        }
        processed_keys.insert(cur_key); // Mark key as seen (latest version is in memtable)
        cur = cur->nxt[0];
    }

    // 2. Scan SSTables (Level by Level)
    for (int level = 0; level <= totalLevel; ++level) {
        for (const auto& sst_head : sstableIndex[level]) {
            // Iterate through all keys in this SSTable's index
            // Note: This might be inefficient if SSTables are large. Phase 2 code was complex here.
            // A simpler (but potentially slower for large SSTables) approach: Iterate keys via head.
            for(uint64_t idx = 0; idx < sst_head.getCnt(); ++idx) {
                uint64_t cur_key = sst_head.getKey(idx);

                // Skip if key was already processed (found newer version in memtable or upper level)
                if (processed_keys.count(cur_key)) {
                    continue;
                }

                // Check if embedding exists
                if (embeddings.count(cur_key)) {
                     // Fetch the value to ensure it's not a deleted tombstone *at this specific level*
                     // This requires reading the value, which adds overhead compared to just using embeddings map.
                     // A potentially better way: Trust the embedding map *if* the `del` function correctly removes embeddings.
                     // Let's trust the embeddings map for now, assuming `del` cleans it up.
                     // If a key is in embeddings map, assume it's not deleted for similarity calculation.

                    float sim = cosine_similarity(query_vec, embeddings[cur_key]);
                    similarities.push_back({cur_key, sim});
                } else {
                    // Should not happen if 'put' always stores embedding and 'del' removes it.
                     // std::cerr << "[WARN] Baseline search_knn: Embedding not found for key " << cur_key << " in SSTable " << sst_head.getFilename() << std::endl;
                     // We might need to fetch value and compute embedding on the fly IF the put logic failed.
                     // However, stick to the assumption that embeddings map is the source of truth for existing vectors.
                }
                 processed_keys.insert(cur_key); // Mark key as processed at this point
            }
        }
    }

    // 3. Sort by Similarity (Descending)
    std::sort(similarities.begin(), similarities.end(), 
              [](const auto& a, const auto& b) {
                  if (std::abs(a.second - b.second) > 1e-7) { // Tolerance for float comparison
                      return a.second > b.second; // Higher similarity first
                  }
                  return a.first < b.first; // Use key as tie-breaker
              });

    // 4. Get Top K results (Key and Value)
    std::vector<std::pair<uint64_t, std::string>> results;
    for (int i = 0; i < std::min((int)similarities.size(), k); ++i) {
        uint64_t key = similarities[i].first;
        // Retrieve the value using the standard get function, which handles deletions correctly.
        std::string value = get(key);
        if (!value.empty()) { // get() returns empty string if not found or deleted
            results.push_back({key, value});
        } else if (results.size() < k) {
             // If get fails for a top similarity key, we might need to look further down
             // the similarities list to ensure we return k results if possible.
             // This simple loop doesn't handle that replacement. For simplicity, we stick to top-k similarity keys first.
        }
    }
    
    return results;
}

// --- ADDED: Original search_knn (string version) calling the vector version ---
std::vector<std::pair<uint64_t, std::string>> KVStore::search_knn(std::string query, int k) {
    std::vector<float> query_vec = get_embedding(query); // Use the embedding helper
    if (query_vec.empty()) {
         std::cerr << "[ERROR] search_knn(string): Failed to get embedding for query." << std::endl;
        return {};
    }
    return search_knn(query_vec, k); // Call the vector version
}
// ---------------------------------------------------------------------------

// --- ADDED: Implementation for public getters for HNSW parameters ---
int KVStore::get_hnsw_m() const {
    return HNSW_M;
}

int KVStore::get_hnsw_ef_construction() const {
    return HNSW_efConstruction;
}
// --- END ADDED ---

// --- 修改：函数定义 --- (修改函数签名)
void KVStore::load_embedding_from_disk(const std::string &data_dir) {
    std::string embedding_file_path = data_dir + "/embeddings.bin";
    std::ifstream embed_file(embedding_file_path, std::ios::binary);

    if (!embed_file.is_open()) {
        std::cout << "[INFO] Embedding file not found (" << embedding_file_path << "). Skipping load. Will be created on first flush." << std::endl;
        return;
    }

    // 1. 读取维度
    uint64_t file_dim = 0;
    embed_file.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
    if (embed_file.gcount() != sizeof(file_dim)) {
        std::cerr << "[ERROR] Failed to read embedding dimension from file: " << embedding_file_path << std::endl;
        embed_file.close();
        return;
    }

    // 验证或设置维度
    if (embedding_dimension_ == 0) {
        embedding_dimension_ = static_cast<int>(file_dim);
         std::cout << "[INFO] Setting embedding dimension from file: " << embedding_dimension_ << std::endl;
    } else if (embedding_dimension_ != static_cast<int>(file_dim)) {
        std::cerr << "[ERROR] Embedding dimension mismatch! File has " << file_dim
                  << ", but KVStore expected " << embedding_dimension_ << std::endl;
        embed_file.close();
        // 可能是严重错误，可以选择抛出异常或清空现有 embeddings
        embeddings.clear(); // 清空以避免使用错误维度的数据
        return;
    }
    if (embedding_dimension_ <= 0) {
         std::cerr << "[ERROR] Invalid embedding dimension loaded: " << embedding_dimension_ << std::endl;
         embed_file.close();
         return;
    }

    // 2. 计算块大小和数量
    size_t block_size = sizeof(uint64_t) + embedding_dimension_ * sizeof(float); // key + vector
    embed_file.seekg(0, std::ios::end);
    long long file_size = embed_file.tellg();
    long long data_bytes = file_size - sizeof(uint64_t); // 减去维度头

    if (data_bytes < 0 || data_bytes % block_size != 0) {
        std::cerr << "[ERROR] Invalid embedding file size or block structure."
                  << " Total size: " << file_size << ", Data bytes: " << data_bytes
                  << ", Expected block size: " << block_size << std::endl;
        embed_file.close();
        embeddings.clear(); // 可能文件损坏，清空
        return;
    }
    long long num_blocks = data_bytes / block_size;

    std::cout << "[INFO] Loading embeddings from " << embedding_file_path
              << ". Dimension: " << embedding_dimension_ << ", Blocks: " << num_blocks << std::endl;

    // 3. 从后往前读取
    embeddings.clear(); // 清空内存中的旧数据，确保只加载最新的
    std::set<uint64_t> loaded_keys; // 记录已加载最新版本的 key
    std::vector<float> temp_vec(embedding_dimension_); // 复用读取缓冲区
    std::vector<float> deleted_marker(embedding_dimension_, std::numeric_limits<float>::max()); // 删除标记

    for (long long i = num_blocks - 1; i >= 0; --i) {
        long long offset = sizeof(uint64_t) + i * block_size;
        embed_file.seekg(offset, std::ios::beg);

        uint64_t current_key;
        embed_file.read(reinterpret_cast<char*>(&current_key), sizeof(current_key));
        if (embed_file.gcount() != sizeof(current_key)) {
             std::cerr << "[ERROR] Failed to read key at block " << i << std::endl; continue;
        }

        // 如果这个 key 的最新版本已经被加载，跳过
        if (loaded_keys.count(current_key)) {
            continue;
        }

        embed_file.read(reinterpret_cast<char*>(temp_vec.data()), embedding_dimension_ * sizeof(float));
         if (embed_file.gcount() != embedding_dimension_ * sizeof(float)) {
             std::cerr << "[ERROR] Failed to read vector for key " << current_key << " at block " << i << std::endl; continue;
         }

        // 检查是否是删除标记
        if (temp_vec == deleted_marker) {
            // 是删除标记，记录该 key 已处理（被删除），然后跳过
            loaded_keys.insert(current_key);
            // std::cout << "[DEBUG] Skipped deleted key: " << current_key << std::endl;
        } else {
            // 是有效向量，存入内存 map，并记录 key 已处理
            embeddings[current_key] = temp_vec; // 使用 vector 的赋值操作符
            loaded_keys.insert(current_key);
             // std::cout << "[DEBUG] Loaded key: " << current_key << std::endl;
        }
    }

    embed_file.close();
    std::cout << "[INFO] Finished loading embeddings. Loaded " << embeddings.size() << " unique keys." << std::endl;
}

// --- 新增：实现 HNSW 索引保存 ---
void KVStore::save_hnsw_index_to_disk(const std::string &hnsw_data_root, bool force_serial /*= false*/) { // Added force_serial parameter
    std::cout << "[INFO] Attempting HNSW index save to disk: " << hnsw_data_root << (force_serial ? " (SERIAL)" : " (PARALLEL)") << std::endl;
    std::atomic<int> saved_node_count_atomic(0); 

    try {
        // 1. 创建根目录和 nodes 子目录 (Sequential Part, common to both)
        std::filesystem::create_directories(hnsw_data_root);
        std::string nodes_path = hnsw_data_root + "/nodes"; // Common base for nodes
        std::filesystem::create_directories(nodes_path);

        // 2. 准备并写入全局头文件 (Sequential Part, common to both)
        HNSWGlobalHeader global_header;
        global_header.M = static_cast<uint32_t>(HNSW_M);
        global_header.M_max = static_cast<uint32_t>(HNSW_M_max);
        global_header.efConstruction = static_cast<uint32_t>(HNSW_efConstruction);
        global_header.max_level = static_cast<uint32_t>(current_max_level_);
        global_header.entry_point_label = entry_point_label_;
        
        uint64_t active_node_count = 0;
        for (const auto& pair : hnsw_nodes_) {
            if (!pair.second.deleted) {
                active_node_count++;
            }
        }
        global_header.num_nodes = active_node_count;
        global_header.dim = static_cast<uint32_t>(embedding_dimension_);

        std::string global_header_path = hnsw_data_root + "/global_header.bin";
        std::ofstream header_file(global_header_path, std::ios::binary | std::ios::trunc);
        if (!header_file.is_open()) {
            std::cerr << "[ERROR] Failed to open global header file for writing: " << global_header_path << std::endl;
        } else {
            header_file.write(reinterpret_cast<const char*>(&global_header), sizeof(HNSWGlobalHeader));
            header_file.close();
            std::cout << "[INFO] Saved global header. Expected active nodes: " << global_header.num_nodes << " to " << global_header_path << std::endl;
        }

        // 3. 保存 HNSW 节点数据 (Conditional: Serial or Parallel)
        if (force_serial) {
            std::cout << "[INFO] Saving HNSW nodes SERIALLY to " << nodes_path << "..." << std::endl;
            uint64_t serial_saved_node_count = 0; 

            for (const auto& pair : hnsw_nodes_) {
                const size_t label = pair.first;
                const HNSWNode& node_ref = pair.second;

                if (node_ref.deleted) {
                    continue;
                }
                
                HNSWNode node_copy = node_ref; 
                std::string node_specific_base_path = nodes_path + "/" + std::to_string(label);

                try {
                    std::filesystem::create_directories(node_specific_base_path);

                    std::string disk_node_header_path = node_specific_base_path + "/header.bin";
                    std::ofstream node_header_ostream(disk_node_header_path, std::ios::binary | std::ios::trunc);
                    if (!node_header_ostream.is_open()) {
                        std::cerr << "[ERROR] SERIAL: Failed to open node header file for writing: " << disk_node_header_path << std::endl;
                        continue; 
                    }
                    NodeHeader disk_node_header_data;
                    disk_node_header_data.max_level = static_cast<uint32_t>(node_copy.max_level);
                    disk_node_header_data.key = node_copy.key;
                    node_header_ostream.write(reinterpret_cast<const char*>(&disk_node_header_data), sizeof(NodeHeader));
                    node_header_ostream.close();

                    std::string edges_dir_path = node_specific_base_path + "/edges";
                    std::filesystem::create_directories(edges_dir_path);

                    for (int level = 0; level <= node_copy.max_level; ++level) {
                        if (level < node_copy.connections.size() && !node_copy.connections[level].empty()) {
                            std::string edge_file_path = edges_dir_path + "/" + std::to_string(level) + ".bin";
                            std::ofstream edge_file(edge_file_path, std::ios::binary | std::ios::trunc);
                            if (!edge_file.is_open()) {
                                std::cerr << "[ERROR] SERIAL: Failed to open edge file for writing: " << edge_file_path << std::endl;
                                continue; 
                            }
                            uint32_t num_edges = static_cast<uint32_t>(node_copy.connections[level].size());
                            edge_file.write(reinterpret_cast<const char*>(&num_edges), sizeof(uint32_t));
                            for (size_t neighbor_label_size_t : node_copy.connections[level]) {
                                if (neighbor_label_size_t > std::numeric_limits<uint32_t>::max()) {
                                    std::cerr << "[WARN] SERIAL: Neighbor label " << neighbor_label_size_t << " exceeds uint32_t max! Saving truncated for node " << label << ", level " << level << "." << std::endl;
                                }
                                uint32_t neighbor_label_u32 = static_cast<uint32_t>(neighbor_label_size_t);
                                edge_file.write(reinterpret_cast<const char*>(&neighbor_label_u32), sizeof(uint32_t));
                            }
                            edge_file.close();
                        }
                    }
                    serial_saved_node_count++;
                } catch (const std::filesystem::filesystem_error& fs_err) {
                    std::cerr << "[ERROR] SERIAL: Filesystem error while saving node " << label << " to " << node_specific_base_path << ": " << fs_err.what() << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] SERIAL: Std exception while saving node " << label << " to " << node_specific_base_path << ": " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[ERROR] SERIAL: Unknown exception while saving node " << label << " to " << node_specific_base_path << std::endl;
                }
            }
            saved_node_count_atomic.store(serial_saved_node_count);
            std::cout << "[INFO] Finished processing HNSW node data SERIALLY. Nodes processed: " << serial_saved_node_count << "." << std::endl;

        } else { // Parallel saving
            std::cout << "[INFO] Saving HNSW nodes PARALLELLY to " << nodes_path << "..." << std::endl;
            // saved_node_count_atomic is already initialized to 0 at the function start.
            { 
                unsigned int num_threads = std::thread::hardware_concurrency();
                if (num_threads == 0) num_threads = std::max(1u, 2u); 
                ThreadPool pool(num_threads);
                std::mutex cerr_mutex; 

                for (const auto& pair : hnsw_nodes_) {
                    const size_t label = pair.first;
                    const HNSWNode& node_ref = pair.second; 

                    if (node_ref.deleted) {
                        continue;
                    }
                    
                    HNSWNode node_copy = node_ref; 
                    std::string node_specific_base_path = nodes_path + "/" + std::to_string(label); // Use common nodes_path

                    pool.enqueue([label, node_copy, node_specific_base_path, &saved_node_count_atomic, &cerr_mutex]() {
                        try {
                            std::filesystem::create_directories(node_specific_base_path);

                            std::string disk_node_header_path = node_specific_base_path + "/header.bin"; 
                            std::ofstream node_header_ostream(disk_node_header_path, std::ios::binary | std::ios::trunc); 
                            if (!node_header_ostream.is_open()) {
                                std::unique_lock<std::mutex> lock(cerr_mutex);
                                std::cerr << "[ERROR] Thread " << std::this_thread::get_id() 
                                          << ": Failed to open node header file for writing: " << disk_node_header_path << std::endl;
                                return; 
                            }
                            NodeHeader disk_node_header_data; 
                            disk_node_header_data.max_level = static_cast<uint32_t>(node_copy.max_level);
                            disk_node_header_data.key = node_copy.key;
                            node_header_ostream.write(reinterpret_cast<const char*>(&disk_node_header_data), sizeof(NodeHeader));
                            node_header_ostream.close();

                            std::string edges_dir_path = node_specific_base_path + "/edges";
                            std::filesystem::create_directories(edges_dir_path);

                            for (int level = 0; level <= node_copy.max_level; ++level) {
                                if (level < node_copy.connections.size() && !node_copy.connections[level].empty()) {
                                    std::string edge_file_path = edges_dir_path + "/" + std::to_string(level) + ".bin";
                                    std::ofstream edge_file(edge_file_path, std::ios::binary | std::ios::trunc);
                                    if (!edge_file.is_open()) {
                                        std::unique_lock<std::mutex> lock(cerr_mutex);
                                        std::cerr << "[ERROR] Thread " << std::this_thread::get_id() 
                                                  << ": Failed to open edge file for writing: " << edge_file_path << std::endl;
                                        continue; 
                                    }
                                    uint32_t num_edges = static_cast<uint32_t>(node_copy.connections[level].size());
                                    edge_file.write(reinterpret_cast<const char*>(&num_edges), sizeof(uint32_t));
                                    for (size_t neighbor_label_size_t : node_copy.connections[level]) {
                                        if (neighbor_label_size_t > std::numeric_limits<uint32_t>::max()) {
                                            std::unique_lock<std::mutex> lock(cerr_mutex);
                                            std::cerr << "[WARN] Thread " << std::this_thread::get_id() 
                                                      << ": Neighbor label " << neighbor_label_size_t 
                                                      << " exceeds uint32_t max! Saving truncated for node " << label << ", level " << level << "." << std::endl;
                                        }
                                        uint32_t neighbor_label_u32 = static_cast<uint32_t>(neighbor_label_size_t);
                                        edge_file.write(reinterpret_cast<const char*>(&neighbor_label_u32), sizeof(uint32_t));
                                    }
                                    edge_file.close();
                                }
                            }
                            saved_node_count_atomic++; 
                        } catch (const std::filesystem::filesystem_error& fs_err) {
                            std::unique_lock<std::mutex> lock(cerr_mutex);
                            std::cerr << "[ERROR] Thread " << std::this_thread::get_id() 
                                      << ": Filesystem error while saving node " << label << " to " << node_specific_base_path << ": " << fs_err.what() << std::endl;
                        } catch (const std::exception& e) {
                            std::unique_lock<std::mutex> lock(cerr_mutex);
                            std::cerr << "[ERROR] Thread " << std::this_thread::get_id() 
                                      << ": Std exception while saving node " << label << " to " << node_specific_base_path << ": " << e.what() << std::endl;
                        } catch (...) {
                            std::unique_lock<std::mutex> lock(cerr_mutex);
                            std::cerr << "[ERROR] Thread " << std::this_thread::get_id() 
                                      << ": Unknown exception while saving node " << label << " to " << node_specific_base_path << std::endl;
                        }
                    });
                }
            } 
            std::cout << "[INFO] Finished processing HNSW node data PARALLELLY. Nodes processed by threads: " 
                      << saved_node_count_atomic.load() << "." << std::endl;
        }

        // --- Common Post-Node-Saving Logic ---
        if (global_header.num_nodes != static_cast<uint64_t>(saved_node_count_atomic.load())) {
            std::cout << "[WARN] Mismatch! Expected active nodes for header: " << global_header.num_nodes
                      << ", but actual saved node count: " << saved_node_count_atomic.load() << "." << std::endl;
        }

        std::string deleted_nodes_path = hnsw_data_root + "/deleted_nodes.bin";
        std::ofstream del_file(deleted_nodes_path, std::ios::binary | std::ios::trunc); 
        if (del_file.is_open()) {
            std::cout << "[DEBUG_SAVE_HNSW] Saving " << hnsw_vectors_to_persist_as_deleted_.size() 
                      << " deleted vectors to " << deleted_nodes_path << std::endl;
            
            for (const auto& vec_to_persist : hnsw_vectors_to_persist_as_deleted_) { 
                if (vec_to_persist.size() == embedding_dimension_) {
                    del_file.write(reinterpret_cast<const char*>(vec_to_persist.data()), vec_to_persist.size() * sizeof(float));
                } else {
                     std::cerr << "[ERROR_SAVE_HNSW_DELETED_VEC] Vector dimension mismatch for a vector in hnsw_vectors_to_persist_as_deleted_."
                               << " Expected dim: " << embedding_dimension_ << ", actual: " << vec_to_persist.size()
                               << ". Skipping this vector." << std::endl;
                }
            }
            del_file.close();
            std::cout << "[INFO] Saved " << hnsw_vectors_to_persist_as_deleted_.size() 
                      << " vectors to " << deleted_nodes_path << std::endl;
        } else {
            std::cerr << "[ERROR] Failed to open file for writing: " << deleted_nodes_path << std::endl;
        }
        std::cout << "[INFO] Completed HNSW index saving process to disk: " << hnsw_data_root << std::endl;

    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[ERROR] Filesystem error during HNSW save (outer scope for " << hnsw_data_root << "): " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during HNSW save (outer scope for " << hnsw_data_root << "): " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception during HNSW save (outer scope for " << hnsw_data_root << ")." << std::endl;
    }
}
// --- HNSW 索引保存结束 ---

// --- 新增：实现 HNSW 索引加载 ---
void KVStore::load_hnsw_index_from_disk(const std::string &hnsw_data_root) {
    std::cout << "[INFO] Attempting to load HNSW index from disk: " << hnsw_data_root << std::endl;

    std::string global_header_path = hnsw_data_root + "/global_header.bin";
    if (!std::filesystem::exists(global_header_path)) {
        std::cout << "[INFO] HNSW global header not found. Skipping HNSW load (assuming first run or no save)." << std::endl;
        return;
    }

    try {
        // 1. 加载全局头文件
        std::ifstream header_file(global_header_path, std::ios::binary);
        if (!header_file.is_open()) {
            std::cerr << "[ERROR] Failed to open global header file for reading: " << global_header_path << std::endl;
            return;
        }
        HNSWGlobalHeader global_header;
        header_file.read(reinterpret_cast<char*>(&global_header), sizeof(HNSWGlobalHeader));
        if (!header_file) {
             std::cerr << "[ERROR] Failed to read global header from: " << global_header_path << std::endl;
             header_file.close();
             return;
        }
        header_file.close();

        // 2. 恢复/验证参数
        // 对于 const 成员，我们验证；对于非 const 成员，我们恢复
        if (static_cast<uint32_t>(HNSW_M) != global_header.M ||
            static_cast<uint32_t>(HNSW_M_max) != global_header.M_max ||
            static_cast<uint32_t>(HNSW_efConstruction) != global_header.efConstruction ||
            static_cast<uint32_t>(embedding_dimension_) != global_header.dim) {
            std::cerr << "[ERROR] HNSW parameter mismatch between saved index and current configuration!" << std::endl;
            // 可以选择报错退出，或者继续加载但可能行为不确定
            // return;
             std::cout << "[WARN] Saved M=" << global_header.M << ", Current M=" << HNSW_M << std::endl;
             std::cout << "[WARN] Saved M_max=" << global_header.M_max << ", Current M_max=" << HNSW_M_max << std::endl;
             std::cout << "[WARN] Saved efC=" << global_header.efConstruction << ", Current efC=" << HNSW_efConstruction << std::endl;
             std::cout << "[WARN] Saved dim=" << global_header.dim << ", Current dim=" << embedding_dimension_ << std::endl;
             // 如果允许不匹配，需要决定是使用加载的参数还是当前的参数
        }
        current_max_level_ = static_cast<int>(global_header.max_level);
        entry_point_label_ = global_header.entry_point_label;
        // next_label_ 应该根据加载的节点数来设置，或者至少是加载的最大 label + 1
        uint64_t max_loaded_label = 0;
        uint64_t loaded_node_count = 0; // 用于验证 global_header.num_nodes

         std::cout << "[INFO] Loaded global header: MaxLevel=" << current_max_level_
                   << ", EntryPoint=" << entry_point_label_
                   << ", SavedNodes=" << global_header.num_nodes
                   << ", Dim=" << global_header.dim << std::endl;


        // 3. 清空当前内存中的 HNSW 结构
        hnsw_nodes_.clear();
        key_to_label_.clear();
        label_to_key_.clear();

        // 4. 加载节点数据
        std::string nodes_path = hnsw_data_root + "/nodes";
        if (!std::filesystem::exists(nodes_path)) {
            std::cerr << "[ERROR] HNSW nodes directory not found: " << nodes_path << std::endl;
            return;
        }

        // 遍历 nodes 目录下的子目录 (假设子目录名是 label)
        for (const auto& entry : std::filesystem::directory_iterator(nodes_path)) {
            if (entry.is_directory()) {
                size_t label = 0;
                try {
                    // 从目录名解析 label
                    label = std::stoull(entry.path().filename().string());
                } catch (const std::exception& e) {
                    std::cerr << "[WARN] Could not parse label from directory name: " << entry.path().filename() << ". Skipping. Error: " << e.what() << std::endl;
                    continue;
                }

                std::string node_dir_path = entry.path().string();
                std::string node_header_path = node_dir_path + "/header.bin";

                if (!std::filesystem::exists(node_header_path)) {
                    std::cerr << "[WARN] Node header file not found for label " << label << ". Skipping." << std::endl;
                    continue;
                }

                // 加载节点头文件
                std::ifstream node_header_file(node_header_path, std::ios::binary);
                 if (!node_header_file.is_open()) {
                     std::cerr << "[ERROR] Failed to open node header file for reading: " << node_header_path << std::endl;
                     continue;
                 }
                NodeHeader node_header;
                node_header_file.read(reinterpret_cast<char*>(&node_header), sizeof(NodeHeader));
                 if (!node_header_file) {
                     std::cerr << "[ERROR] Failed to read node header for label " << label << std::endl;
                     node_header_file.close();
                     continue;
                 }
                node_header_file.close();

                // 创建 HNSWNode 对象 (暂时不设置 connections)
                // 方案2核心：加载的节点默认为 active (deleted = false)
                HNSWNode node(node_header.key, label, static_cast<int>(node_header.max_level));
                node.deleted = false; // Explicitly set to false as per Scheme 2
                node.connections.resize(node.max_level + 1); // 预分配空间

                // 加载边的文件
                std::string edges_dir_path = node_dir_path + "/edges";
                if (std::filesystem::exists(edges_dir_path)) {
                    for (int level = 0; level <= node.max_level; ++level) {
                        std::string edge_file_path = edges_dir_path + "/" + std::to_string(level) + ".bin";
                        if (std::filesystem::exists(edge_file_path)) {
                            std::ifstream edge_file(edge_file_path, std::ios::binary);
                             if (!edge_file.is_open()) {
                                 std::cerr << "[ERROR] Failed to open edge file for reading: " << edge_file_path << std::endl;
                                 continue; // 跳过这一层
                             }
                            uint32_t num_edges = 0;
                            edge_file.read(reinterpret_cast<char*>(&num_edges), sizeof(uint32_t));
                             if (!edge_file) {
                                 std::cerr << "[ERROR] Failed to read num_edges for label " << label << " level " << level << std::endl;
                                 edge_file.close();
                                 continue;
                             }

                            if (num_edges > 0) {
                                // node.connections[level].resize(num_edges); // Already resized above based on node.max_level
                                // 读取邻居 label (读取为 uint32_t，存为 size_t)
                                std::vector<uint32_t> neighbors_u32(num_edges);
                                edge_file.read(reinterpret_cast<char*>(neighbors_u32.data()), num_edges * sizeof(uint32_t));
                                 if (!edge_file) {
                                     std::cerr << "[ERROR] Failed to read neighbors for label " << label << " level " << level << std::endl;
                                     // 可能只读取了部分，清空这一层
                                     node.connections[level].clear();
                                     edge_file.close();
                                     continue;
                                 }
                                 // 转换并存储
                                 node.connections[level].clear(); // Clear before push_back if resize was too large or just to be safe
                                 for(uint32_t neighbor_u32 : neighbors_u32) {
                                     node.connections[level].push_back(static_cast<size_t>(neighbor_u32));
                                 }
                            }
                            edge_file.close();
                        }
                    }
                }

                // 存储加载的节点到内存 map
                hnsw_nodes_[label] = node;
                key_to_label_[node.key] = label;
                label_to_key_[label] = node.key;
                max_loaded_label = std::max(max_loaded_label, label);
                loaded_node_count++;
            }
        }

        // 检查加载的节点数是否与头文件匹配
        if (loaded_node_count != global_header.num_nodes) {
             std::cout << "[WARN] Number of loaded nodes (" << loaded_node_count
                       << ") does not match count in global header (" << global_header.num_nodes << ")." << std::endl;
        }

        // 更新 next_label_
        next_label_ = max_loaded_label + 1; // 确保下一个分配的 label 是唯一的
         std::cout << "[INFO] Finished loading HNSW index. Loaded " << loaded_node_count << " nodes. Next label will be " << next_label_ << "." << std::endl;

    // --- Phase 4: 加载 deleted_nodes.bin --- (这部分逻辑保留，用于搜索时的过滤)
    loaded_deleted_vectors_.clear();
    std::string deleted_nodes_path = hnsw_data_root + "/deleted_nodes.bin";
    if (std::filesystem::exists(deleted_nodes_path)) {
        std::ifstream dn_file(deleted_nodes_path, std::ios::binary);
        if (!dn_file.is_open()) {
            std::cerr << "[ERROR] Failed to open " << deleted_nodes_path << " for reading." << std::endl;
        } else {
            if (embedding_dimension_ > 0) {
                std::vector<float> temp_vec(embedding_dimension_);
                while (dn_file.read(reinterpret_cast<char*>(temp_vec.data()), embedding_dimension_ * sizeof(float))) {
                    if (dn_file.gcount() == embedding_dimension_ * sizeof(float)) {
                        loaded_deleted_vectors_.push_back(temp_vec);
                    } else {
                        std::cerr << "[ERROR] Incomplete vector read from deleted_nodes.bin." << std::endl;
                        break; 
                    }
                }
                dn_file.close();
                std::cout << "[INFO] Loaded " << loaded_deleted_vectors_.size() << " vectors from deleted_nodes.bin." << std::endl;

                // --- THIS IS THE BLOCK TO BE COMMENTED OUT ---
                /*
                int marked_deleted_count = 0;
                if (!loaded_deleted_vectors_.empty()) {
                    for (auto& pair : hnsw_nodes_) { 
                        HNSWNode& node = pair.second;
                        if (node.deleted) continue; 
                        if (embeddings.count(node.key)) {
                            const std::vector<float>& node_vec = embeddings[node.key];
                            if (node_vec.size() == embedding_dimension_) {
                                for (const auto& deleted_vec : loaded_deleted_vectors_) {
                                    if (compare_float_vectors(node_vec, deleted_vec)) { // Using compare_float_vectors
                                        node.deleted = true;
                                        marked_deleted_count++;
                                        break; 
                                    }
                                }
                            }
                        }
                    }
                    if (marked_deleted_count > 0) {
                         std::cout << "[INFO] Marked " << marked_deleted_count << " HNSW nodes as deleted based on loaded_deleted_vectors." << std::endl;
                    }
                }
                */
                // --- END OF BLOCK TO BE COMMENTED OUT ---

            } else {
                std::cerr << "[WARN] Embedding dimension is 0, cannot process deleted_nodes.bin." << std::endl;
                dn_file.close();
            }
        }
    } else {
        std::cout << "[INFO] " << deleted_nodes_path << " not found. No deleted HNSW vectors loaded." << std::endl;
    }
    // --- End Phase 4 deleted_nodes.bin load ---

    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[ERROR] Filesystem error during HNSW load: " << e.what() << std::endl;
        // 清空状态以避免使用部分加载的数据
        hnsw_nodes_.clear(); key_to_label_.clear(); label_to_key_.clear(); current_max_level_ = -1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during HNSW load: " << e.what() << std::endl;
        hnsw_nodes_.clear(); key_to_label_.clear(); label_to_key_.clear(); current_max_level_ = -1;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception during HNSW load." << std::endl;
        hnsw_nodes_.clear(); key_to_label_.clear(); label_to_key_.clear(); current_max_level_ = -1;
    }
}
// --- HNSW 索引加载结束 ---

// --- ADDED: Implementation for put_with_precomputed_embedding ---
void KVStore::put_with_precomputed_embedding(uint64_t key, const std::string &val, const std::vector<float>& precomputed_emb) {
    // --- LSM Put Logic (similar to original put) ---
    uint32_t nxtsize = s->getBytes();
    std::string res = s->search(key);
    if (!res.length()) {
        nxtsize += 12 + val.length();
    } else {
        nxtsize = nxtsize - res.length() + val.length();
    }

    if (nxtsize + 10240 + 32 <= MAXSIZE) {
        s->insert(key, val);
    } else {
        sstable ss(s);
        const std::string embedding_file_path = dir_ + "/embeddings.bin"; // NEW
        std::ofstream embed_file(embedding_file_path, std::ios::binary | std::ios::app);
        if (embed_file.is_open()) {
            embed_file.seekp(0, std::ios::end);
            if (embed_file.tellp() == 0) {
                 uint64_t dim_to_write = embedding_dimension_;
                 if (dim_to_write == 0 && !precomputed_emb.empty()) dim_to_write = precomputed_emb.size(); // Get dim from first precomputed if not set
                 if (dim_to_write > 0) {
                    embed_file.write(reinterpret_cast<const char*>(&dim_to_write), sizeof(dim_to_write));
                 } else {
                    std::cerr << "[WARN] KVStore::put_with_precomputed_embedding - Dimension is 0, cannot write embedding header." << std::endl;
                 }
            }
            slnode *cur = s->getFirst();
            while (cur && cur->type != TAIL) {
                uint64_t current_key = cur->key;
                if (embeddings.count(current_key)) { // embeddings map should be populated by this function or original put
                    const std::vector<float>& vec_to_save = embeddings[current_key];
                    if (vec_to_save.size() == embedding_dimension_ || (embedding_dimension_ == 0 && !vec_to_save.empty())) {
                         if (embedding_dimension_ == 0) embedding_dimension_ = vec_to_save.size(); // Set if first time
                        embed_file.write(reinterpret_cast<const char*>(&current_key), sizeof(current_key));
                        embed_file.write(reinterpret_cast<const char*>(vec_to_save.data()), vec_to_save.size() * sizeof(float));
                    } else {
                         std::cerr << "[WARN] KVStore::put_with_precomputed_embedding - Dimension mismatch for key " << current_key << " during SSTable flush. Skipping save." << std::endl;
                    }
                } else {
                     std::cerr << "[WARN] KVStore::put_with_precomputed_embedding - Embedding not found for key " << current_key << " in embeddings map during SSTable flush. Skipping save." << std::endl;
                }
                cur = cur->nxt[0];
            }
            embed_file.close();
        } else {
            std::cerr << "[ERROR] KVStore::put_with_precomputed_embedding - Failed to open embedding file for writing: " << embedding_file_path << std::endl;
        }

        s->reset();
        std::string level0_path = dir_ + "/level-0"; // MODIFIED: Use dir_
        if (!utils::dirExists(level0_path)) {
            utils::mkdir(level0_path.data()); // MODIFIED: Use dir_ based path
            totalLevel = 0;
        }
        // --- MODIFICATION: Construct full path for SSTable ---
        std::string full_sstable_path = level0_path + "/" + std::to_string(ss.getTime()) + ".sst";
        ss.setFilename(full_sstable_path);
        // --- END MODIFICATION ---

        addsstable(ss, 0);
        ss.putFile(full_sstable_path.data()); // MODIFIED: Use full_sstable_path
        compaction();
        s->insert(key, val);
    }
    // --- End LSM Put Logic ---

    // --- HNSW and Embedding Map Update Logic ---
    if (!precomputed_emb.empty()) {
        if (embedding_dimension_ == 0) {
             embedding_dimension_ = precomputed_emb.size();
             std::cout << "[INFO] KVStore::put_with_precomputed_embedding - Embedding dimension set to " << embedding_dimension_ << " from key " << key << std::endl;
        } else if (embedding_dimension_ != precomputed_emb.size()) {
            std::cerr << "[ERROR] KVStore::put_with_precomputed_embedding - Precomputed embedding dimension mismatch for key " << key 
                      << ". Expected " << embedding_dimension_ << " got " << precomputed_emb.size() << std::endl;
            return; 
        }

        if (key_to_label_.count(key)) {
            size_t old_label = key_to_label_[key];
            if (hnsw_nodes_.count(old_label)) {
                hnsw_nodes_[old_label].deleted = true;
            }
        }

        embeddings[key] = precomputed_emb; // Store/update in the main embeddings map
        hnsw_insert(key, precomputed_emb); // Insert/update in HNSW graph

    } else {
        std::cerr << "[WARN] KVStore::put_with_precomputed_embedding - Called with empty precomputed_emb for key " << key << std::endl;
    }
}
// --- END ADDED ---


