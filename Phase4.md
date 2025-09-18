# Project阶段4: Persistent Vector DB

## Introduction

为了避免在此阶段以及之后的阶段中频繁地进行embedding向量操作

首先，你需要实现将 `embedding` 的向量保存到磁盘的操作并且可以从磁盘中读取 `embedding` 的向量

然后，你需要支持 `HNSW` 索引结构的删除和修改操作

最后，你需要持久化 `HNSW` 结构，主要是将构建好的索引保存到磁盘，以便在后续的查询中复用，避免每次启动时都重新构建索引。

## Task

本节介绍本阶段需要完成的任务。

### 将 `embedding` 的向量保存到磁盘

在每次 kv 对所在的 `skip list` 被写入磁盘时，需要将 `embedding` 的向量保存到磁盘

你需要实现的是实现方式是将 `key` -> `embedding vector` 的映射关系保存到磁盘，在 `lsmtree` 中通过 `key` 查询 `sentence`，然后得到 `sentence` -> `embedding vector` 的映射关系

类似于 `lsmtree` ，数据以二进制形式保存，只需要不断追加而不需要修改，每次都将新的 `key` 和 `embedding vector` 追加到文件末尾

所以，每个数据块的格式如下：

```c++
struct DataBlock {
    uint64_t key;
    std::vector<float> embedding; // size is equal to the dimension(768 in this model) 
};
```

为了更方便的通过偏移量读取每个 `embedding vector`，需要将每个 `embedding vector` 的 `dimension` 记录在 **文件开头** 中

在本次 project 中，我们选取的模型是 `nomic-embed-text-v1.5.Q8_0.gguf`，通过读取这个模型的信息，可以知道每个 embedding vector 的 `dimension` 为 **768** (也可以通过一次 embedding 操作，得到 embedding vector 的 `dimension`)

所以，每个数据块的大小为 `8 + 768 * 4 = 3080` bytes

#### 文件布局示意图

以 `dim=768`为例，文件布局示意图如下：

```
+------------------+------------------+------------------+------------------+-----+------------------+------------------+
|                  |                  |                  |                  |     |                  |                  |
| uint64_t         | uint64_t         | float[768]       | uint64_t         |     | uint64_t         | float[768]       |
| (embedding dim)  |   (key)          | (vector data)    |   (key)          | ... |   (key)          | (vector data)    |
|                  |                  |                  |                  |     |                  |                  |
+------------------+------------------+------------------+------------------+-----+------------------+------------------+
```

#### 删除操作

在删除操作中，需要插入一个特殊的embedding vector，这个vector为 `std::vector<float>(dim, std::numeric_limits<float>::max())`，其中 `dim` 为 `embedding vector` 的维度
即所有维度都为 `std::numeric_limits<float>::max()`


#### 修改操作

在修改操作中，只需要向文件中追加一个 `(key, embedding vector)` 即可，不需要修改文件中的其他内容

### 从磁盘加载 `embedding` 的向量

你需要实现函数 `void KVStore::load_embedding_from_disk(const std::string &data_root)` ，系统启动时，从磁盘加载 `embedding` 的向量，以下是加载的流程

1. 读取第一个 `uint64_t` ，得到 `embedding vector` 的维度 `dim` 
2. 得到第 `i` 个数据块的位置是 `8 + (8 + 4 * dim) * i`
3. 因为我们是 **append** 写入的，所以只需要从文件末尾开始，根据数据块的位置，读取每个数据块
4. 从后往前读到的第一个 `(key, embedding vector)` 即为 当前 `key` 对应的 `embedding vector`
5. 如果读到的 `embedding vector` 所有维度都为 `std::numeric_limits<float>::max()`，则说明当前 `key` 对应的 `embedding vector` 被删除，需要跳过

### 支持 `HNSW` 索引结构的删除操作

`HNSW` 算法本身并不支持删除操作，所以需要实现的是最简单的 `Lazy Delete` 操作，即在删除节点时，即不立即从 `HNSW` 中删除节点，而是将节点标记为删除，在查询时忽略被标记为删除的节点。

为了支持这一算法，你需要创建一个 `deleted_nodes` 的 `set` 结构，用于记录被删除的节点

```c++
std::vector<std::vector<float>> deleted_nodes;
```

其中 `deleted_nodes` 中的元素为被删除节点的 `向量数据`

以下是Phase3中描述的 **Query** 过程，我们需要简单的修改，忽略被标记为删除的节点

```md
### **Query**
本小节介绍查询目标节点的相邻节点的算法。查询过程与插入过程相近，query过程同样分为两步：
* 自顶层向第1层逐层搜索,每层寻找当前层与目标节点q最近邻的1个点赋值到集合W，然后从集合W中选择最接近q的点作为下一层的搜索入口点。
* 假设要查找的是最近的k个节点。接着在第0层中，查找与目标节点q临近的efConstruction个节点，其中选取k个最接近q的节点作为最终结果。注意：在第0层中，查找与目标节点q临近的efConstruction个节点时，可能会找到比k个更多的节点，因此需要对这些节点进行排序，选取前k个最接近q的节点作为最终结果。
```

上述内容中，需要选取k个最接近q，且不在 `deleted_nodes` 中的节点作为最终结果

### 支持 `HNSW` 索引结构的修改操作

在进行修改操作时，只需要将原始节点标记为删除

### 将 `HNSW` 的索引结构保存到磁盘

你需要实现函数 `void KVStore::save_hnsw_index_to_disk(const std::string &hnsw_data_root)`，将 `HNSW` 的索引结构保存到磁盘
其中 `hnsw_data_root` 为保存 `HNSW` 索引结构的根目录

- 需要给每个节点一个全局唯一的 `id`，然后根据 `id` 将节点保存到磁盘，为了方便，我们使用 **自增** 的 `id`
- 需要保存 `HNSW` 的参数，包括 `M`，`M_max`，`efConstruction`，`m_L`，`max_level`，`num_nodes`，`dim`
- 需要保存每个节点的向量数据和邻接表数据

#### 整体架构

```
/hnsw_data_root/
  ├── global_header.bin   # 全局参数文件（同原HNSWHeader结构）
  ├── deleted_nodes.bin   # 被删除的节点数据
  ├── nodes/              # 节点数据存储目录
  │   ├── 0/              # 节点0的数据
  │   │   ├── header.bin  # 向量数据（float32数组）
  │   │   └── edges/      # 邻接表目录
  │   │       ├── 3.bin   # 第3层邻接表
  │   │       ├── 2.bin   # 第2层邻接表
  │   │       └── ...     # 其他存在的层级
  │   ├── 1/              # 节点1的数据
  │   └── ...             # 其他节点
```

#### global_header.bin

```c++
struct HNSWGlobalHeader {
    uint32_t M;                // 参数
    uint32_t M_max;            // 参数
    uint32_t efConstruction;   // 参数
    uint32_t m_L;              // 参数
    uint32_t max_level;        // 全图最高层级
    uint32_t num_nodes;        // 节点总数
    uint32_t dim;              // 向量维度
};
```

#### deleted_nodes.bin

```c++
std::vector<std::vector<float>> deleted_nodes;
```

以 `dim=768`为例，文件布局示意图如下：

```
+------------------+------------------+-----+------------------+------------------+
|                  |                  |     |                  |                  |
| float[768]       | float[768]       |     | float[768]       | float[768]       |
| (vector data)    |   (vector data)  | ... |   (vector data)  | (vector data)    |
|                  |                  |     |                  |                  |
+------------------+------------------+-----+------------------+------------------+
```

#### 单个节点结构

```
├── 0/              # 节点0的数据
│   ├── header.bin  # 节点参数文件
│   └── edges/      # 邻接表目录
│       ├── 0.bin   # 第0层邻接表
│       ├── 1.bin   # 第1层邻接表
│       └── ...     # 其他存在的层级
```

##### header.bin

```c++
struct NodeHeader {
    uint32_t max_level;               // 当前节点所在的层级
    uint64_t key_of_embedding_vector; // 节点中的向量数据，但是用vector对应的key表示
};
```

##### i.bin (i 为层级)

```c++
struct EdgeFile {
    uint32_t num_edges;        // 邻居数量（≤M_max）
    uint32_t neighbors[];      // 邻居ID数组
};
```

### 从磁盘加载 `HNSW` 的索引结构

你需要实现函数 `void KVStore::load_hnsw_index_from_disk(const std::string &hnsw_data_root)`，从磁盘加载 `HNSW` 的索引结构
其中 `hnsw_data_root` 为保存 `HNSW` 索引结构的根目录

系统启动时，需要从磁盘加载 `HNSW` 的索引结构，以下是加载的流程：

1. 读取 `global_header.bin`，获取 `HNSW` 的参数
2. 按照从id从小到大的顺序读取 `nodes/` 目录下的所有节点的向量数据和邻接表数据
3. 将所有节点数据和邻接表数据加载到内存中

**作为一个project，你只需要完成要求的接口以保证测试用例能够正常运行，我们不对你的实现细节要求做过多限制**

## Test

### Vector_Persistent_Test

在实现完成后，请uncomment `Vector_Persistent_Test` 和 Phase2 中的 `load_embedding_from_disk` 函数，并运行测试用例

测试会插入一些数据，然后保存到磁盘，然后重启数据库，重新从磁盘加载数据，然后查询数据

### HNSW_Delete_Test

测试会一些数据，然后删除一些数据，如果仍然可以查询到被删除的数据，则测试失败

HNSW 的 `Put` 我们不进行测试，`Put` 是依赖于 `Delete` 操作的，如果 `Delete` 操作没有问题， `Put` 操作也不会有问题

### HNSW_Persistent_Test

在实现完成后，请分别uncomment `HNSW_Persistent_Test` Phase1 和 Phase2 中的 `save_hnsw_index_to_disk` 和 `load_hnsw_index_from_disk` 函数，并运行测试用例

测试会插入一些数据，然后保存到磁盘，然后重启数据库，重新从磁盘加载数据，然后查询数据

输出为 `accept rate`，即 `HNSW` 的 `accept rate`

## Report

你需要在报告中附上你的测试结果，以及完成过程中遇到的问题、解决方案和思考等。

### 思考题

如果让你设计以下内容
1. 持久化 `embedding` 的向量，你会如何设计？
2. 持久化 `HNSW` 的索引结构，你会如何设计？
3. 支持 `HNSW` 的删除操作和修改操作，你会如何设计？

## Bouns

[embedding_100k.txt](https://pan.sjtu.edu.cn/web/share/712643d6ba3a9b36d5b0db599a3185e4)

格式：

$$[\alpha_{i_0 j_0}, \alpha_{i_0 j_1}, \alpha_{i_0 j_2}, ..., \alpha_{i_0 j_{767}}]$$
$$[\alpha_{i_1 j_0}, \alpha_{i_1 j_1}, \alpha_{i_1 j_2}, ..., \alpha_{i_1 j_{767}}]$$
$$[\alpha_{i_2 j_0}, \alpha_{i_2 j_1}, \alpha_{i_2 j_2}, ..., \alpha_{i_2 j_{767}}]$$
...
$$[\alpha_{i_{99999} j_0}, \alpha_{i_{99999} j_1}, \alpha_{i_{99999} j_2}, ..., \alpha_{i_{99999} j_{767}}]$$

[cleaned_text_100k.txt](https://pan.sjtu.edu.cn/web/share/56e7272c6a632630ba50939f70817dba)

格式：

```
sentence0
sentence1
...
sentence99999
```

以上是 `embedding_100k.txt` 和 `cleaned_text_100k.txt` 的分享链接，先从这里把数据下载下来

其中`cleaned_text_100k.txt` 是 `sentence` 的文本，`embedding_100k.txt` 是 `embedding` 的向量，包含 **100000** 条数据

语句和向量一一对应，即 `cleaned_text_100k.txt` 中的第 `i` 行语句的 embedding 向量是 `embedding_100k.txt` 中的第 `i` 行

你可以修改 KVStore 的 `put` 函数，绕过 embedding 过程，通过语句直接找到对应的向量，而不需要计算 embedding

加载数据后，你需要尝试持久化这个大数据集，并对这个大数据集建立 `HNSW` 索引结构

我们会在之后的Phase中使用这个大数据集进行测试
