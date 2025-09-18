// 修复后的函数代码，删除了try块外部对global_header的引用

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

        // 添加详细日志
        std::cout << "[DEBUG_LOAD_HNSW] Loaded parameters: M=" << global_header.M 
                  << ", M_max=" << global_header.M_max 
                  << ", efConstruction=" << global_header.efConstruction
                  << ", max_level=" << global_header.max_level
                  << ", num_nodes=" << global_header.num_nodes
                  << ", dim=" << global_header.dim << std::endl;

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

        // 加载完成后验证
        std::cout << "[DEBUG_LOAD_HNSW] After loading: current_max_level_=" << current_max_level_
                  << ", entry_point_label_=" << entry_point_label_
                  << ", next_label_=" << next_label_
                  << ", hnsw_nodes_.size()=" << hnsw_nodes_.size() << std::endl;

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
    
    // 删除原末尾代码（已移到try块内部）
} 