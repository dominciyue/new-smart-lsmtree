#pragma once

#include "kvstore_api.h"
#include "skiplist.h"
#include "sstable.h"
#include "sstablehead.h"

#include <map>
#include <set>
#include <vector>
#include <cmath>       // For std::sqrt, std::log
#include <random>      // For level generation
#include <queue>       // For priority_queue in search
#include <unordered_set> // For visited nodes in search
#include <limits>      // For std::numeric_limits
#include <algorithm>   // For std::max, std::min, std::sort
#include <chrono>      // For timing
#include <memory>      // For std::unique_ptr if needed elsewhere, though not for HNSW now

// --- Phase 3: HNSW 自定义实现所需结构 ---
struct HNSWNode {
    uint64_t key;                    // 对应的 KVStore key
    size_t label;                    // 在 HNSW 图中的唯一标识符
    int max_level;                   // 该节点存在的最高层级 (从 0 开始)
    std::vector<std::vector<size_t>> connections; // connections[i] 存储第 i 层邻居的 label
    bool deleted = false;            // 懒删除标记

    // 构造函数 (示例)
    HNSWNode(uint64_t k, size_t l, int lvl) : key(k), label(l), max_level(lvl) {
        connections.resize(lvl + 1); // 分配层级对应的连接列表空间
    }
    HNSWNode() = default; // 允许默认构造
};

// 用于优先队列存储 (距离, 标签) - 最小堆（距离越小越优先）
using HNSWHeapItem = std::pair<float, size_t>;
struct MinHNSWHeapComparer {
    bool operator()(const HNSWHeapItem& a, const HNSWHeapItem& b) const {
        return a.first > b.first; // 距离小的优先
    }
};
// 用于优先队列存储 (距离, 标签) - 最大堆（距离越大越优先，用于维护固定大小的最近邻列表）
struct MaxHNSWHeapComparer {
     bool operator()(const HNSWHeapItem& a, const HNSWHeapItem& b) const {
        return a.first < b.first; // 距离大的优先
    }
};

// --------------------------------------------

class KVStore : public KVStoreAPI {
    // You can add your implementation here
private:
    std::string dir_; // Added to store the data directory path
    skiplist *s = new skiplist(0.5); // memtable
    // std::vector<sstablehead> sstableIndex;  // sstable的表头缓存

    std::vector<sstablehead> sstableIndex[15]; // the sshead for each level

    int totalLevel = -1; // 层数

    // 添加嵌入向量存储
    std::map<uint64_t, std::vector<float>> embeddings; // 存储key对应的value的向量表示

    // --- Phase 3: HNSW 自定义实现所需成员 ---
    std::map<size_t, HNSWNode> hnsw_nodes_; // 存储所有 HNSW 节点 (label -> Node)
    std::map<uint64_t, size_t> key_to_label_; // KVStore key -> HNSW label
    std::map<size_t, uint64_t> label_to_key_; // HNSW label -> KVStore key
    size_t next_label_ = 0;
    size_t entry_point_label_ = 0; // HNSW 图的入口点 label
    int current_max_level_ = -1;   // 当前 HNSW 图的最高层级 (初始化为 -1 表示空图)
    int embedding_dimension_ = 0;  // 向量维度

    // --- Phase 4: HNSW 删除持久化所需成员 ---
    // std::set<uint64_t> keys_marked_for_hnsw_deletion_; // 存储被del标记的HNSW key，用于写入deleted_nodes.bin
    std::vector<std::vector<float>> hnsw_vectors_to_persist_as_deleted_; // Stores actual vectors of HNSW nodes that were deleted
    std::vector<std::vector<float>> loaded_deleted_vectors_; // 从deleted_nodes.bin加载的向量，用于查询时过滤
    // ------------------------------------------

    // HNSW 参数 (根据 README-phase3.md 推荐值或自行调整)
    const int HNSW_M = 10;             // 每层连接数
    const int HNSW_M_max = 20;         // 每层最大连接数 (通常是 M 的 2 倍左右)
    const int HNSW_efConstruction = 100; // 构建时候选列表大小 (原为 40)
    const double HNSW_m_L = 1.0 / std::log(static_cast<double>(HNSW_M)); // 层数选择参数

    // 随机数生成器 (用于层级选择)
    std::mt19937 rng_{std::random_device{}()};

    // --- Phase 3: HNSW 内部辅助函数声明 ---
    float calculate_distance(const std::vector<float>& v1, const std::vector<float>& v2);
    int get_random_level();
    std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer>
        search_layer_internal(size_t entry_point_label,
                              const std::vector<float>& query_vec,
                              int target_level,
                              int ef, // ef 控制搜索范围/返回数量
                              bool limited_search = false); // true表示只找最近的1个(用于高层)
    std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer>
        search_base_layer(size_t entry_point_label, const std::vector<float>& query_vec, int efSearch);
    std::vector<size_t> select_neighbors(
            std::priority_queue<HNSWHeapItem, std::vector<HNSWHeapItem>, MinHNSWHeapComparer>& candidates,
            int M);
    void hnsw_insert(uint64_t key, const std::vector<float>& vec);
    void prune_connections(size_t node_label, int level, int max_conn); // Helper for M_max pruning

    // --- ADDED: Custom float vector comparison with tolerance ---
    static bool compare_float_vectors(const std::vector<float>& v1, const std::vector<float>& v2, float epsilon = 1e-1f);
    // --- END ADDED ---

public:
    KVStore(const std::string &dir, const std::string &hnsw_index_path = "");

    ~KVStore();

    void put(uint64_t key, const std::string &s) override;

    std::string get(uint64_t key) override;

    bool del(uint64_t key) override;

    void reset() override;

    void scan(uint64_t key1, uint64_t key2, std::list<std::pair<uint64_t, std::string>> &list) override;

    // 添加KNN搜索函数
    std::vector<std::pair<uint64_t, std::string>> search_knn(std::string query, int k);
    std::vector<std::pair<uint64_t, std::string>> search_knn(const std::vector<float>& query_vec, int k);

    // 增加search_knn_hnsw函数声明
    std::vector<std::pair<uint64_t, std::string>> search_knn_hnsw(std::string query, int k);
    std::vector<std::pair<uint64_t, std::string>> search_knn_hnsw(const std::vector<float>& query_vec, int k);
    std::vector<std::pair<uint64_t, std::string>> search_knn_hnsw(const std::vector<float>& query_vec, int k, bool is_string_query, const std::string& query_text);
    
    // 向量处理函数
    std::vector<float> get_embedding(const std::string& text);

    // HNSW参数获取函数
    int get_hnsw_m() const;
    int get_hnsw_ef_construction() const;

    // 持久化函数
    void load_embedding_from_disk(const std::string &data_dir);
    void save_hnsw_index_to_disk(const std::string &hnsw_data_root, bool force_serial = false);
    void load_hnsw_index_from_disk(const std::string &hnsw_data_root);

    void compaction();

    void delsstable(std::string filename);  // 从缓存中删除filename.sst， 并物理删除
    void addsstable(sstable ss, int level); // 将ss加入缓存

    std::string fetchString(std::string file, int startOffset, uint32_t len);

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b); // Phase 2 已有

    // 添加用于大规模预计算嵌入的功能
    void put_with_precomputed_embedding(uint64_t key, const std::string &s, const std::vector<float>& precomputed_emb);
};
