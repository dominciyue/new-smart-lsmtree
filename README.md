# smart_lsm-tree

高性能 KV 存储与持久化索引实验项目（C++/CMake）。项目包含：
- LSM-Tree 相关组件（`skiplist`、`sstable`、BloomFilter 等）
- 持久化与并行读取、MapReduce 风格并行处理
- 向量/图索引实验（HNSW 等测试用例）
- 完整的单元/阶段性测试工程（`test/`）

> 本仓库主要面向课程/实验与工程实践的结合，适合学习与二次开发。

---

## 环境依赖
- CMake ≥ 3.16
- C++17 编译器
  - Windows: MSVC (Visual Studio 2019/2022) 或 MinGW + Ninja
  - Linux/macOS: GCC/Clang
- 可选：Ninja（更快的构建器）

克隆仓库（HTTPS 示例）：
```bash
git clone https://github.com/yourname/smart_lsm-tree.git
cd smart_lsm-tree
```

如果你使用的是当前项目目录（已存在），可以跳过克隆。

---

## 快速开始

### Windows（MSVC）
```powershell
# 生成 VS 解决方案（x64）
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
# Debug/Release 二选一
cmake --build build --config Release

# 运行测试（可选）
ctest --test-dir build -C Release -V
```

### 跨平台快速构建（Ninja）
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -V
```

构建产物默认在 `build/` 目录中。若你在 Windows 使用多配置生成器（如 Visual Studio），请通过 `--config Debug|Release` 指定配置。

---

## 依赖获取（第三方与子模块）
本项目的部分功能依赖 `third_party/llama.cpp`。为避免仓库过大，推荐使用 Git 子模块托管：

使用者在克隆后执行任一方式完成依赖拉取：
```bash
# 方式 A：一次性递归克隆
git clone --recurse-submodules https://github.com/<you>/new-smart-lsmtree.git

# 方式 B：已克隆仓库后补拉子模块
git submodule update --init --recursive
```

维护者在新增子模块时：
```bash
git submodule add https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp
git commit -m "chore: add llama.cpp as submodule"
git push
```

更新子模块到指定版本：
```bash
cd third_party/llama.cpp
git fetch
git checkout <tag-or-commit>
cd ../..
git add third_party/llama.cpp
git commit -m "chore: bump llama.cpp submodule"
git push
```

> 如果你不想使用子模块，也可以在构建时通过 CMake 的 `FetchContent` 拉取上游，但子模块更便于版本固定与离线开发。

---

## 模型下载与放置
出于体积限制，模型文件不随仓库分发。请从 Releases 或外部链接下载后放置到 `model/` 目录：

```text
smart_lsm-tree/
  model/
    nomic-embed-text-v1.5.Q8_0.gguf
```

`.gitignore` 已忽略 `model/` 与常见大模型后缀；若团队协作需要，也可以改用 Git LFS 托管。

---

## 目录结构
```
smart_lsm-tree/
  bloom.cpp|h               # BloomFilter 实现
  skiplist.cpp|h            # 内存跳表（MemTable）
  sstable.cpp|h             # SSTable 读写
  sstablehead.cpp|h         # SSTable 头部与元数据
  kvstore.cc|h              # KV Store 对外接口/封装
  kvstore_api.h             # KV API 说明
  persistence.cc            # 持久化相关逻辑
  readfile-parallel.cc      # 并行文件读取示例
  mapreduce-parallel.cc     # MapReduce 风格并行示例
  threadpool.cc             # 线程池实现
  utils.h                   # 常用工具
  test/                     # 测试工程（CMake、用例）
  Phase4.md / Phase5.md     # 阶段性说明文档
  main.tex                  # LaTeX 文档（论文/报告）
  CMakeLists.txt            # CMake 入口
  .clang-format             # 代码风格
  .gitignore                # 忽略规则
```

> 仓库中还包含若干 `*_Test.cpp`/`*_Persistent_Test_Phase*.cpp` 作为阶段性/专项测试。

---

## 常用命令
- 配置生成（Visual Studio）：
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```
- 配置生成（Ninja 单配置）：
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```
- 编译：
```bash
cmake --build build --config Release
```
- 运行测试：
```bash
ctest --test-dir build -C Release -V
```

---

## 代码风格
项目根目录提供 `.clang-format`。建议在提交前格式化代码：
```bash
clang-format -i path/to/file.cpp
```
或在 IDE 中开启保存自动格式化。

---

## FAQ
- 构建目录太多二进制文件？
  - 已在 `.gitignore` 中忽略 `build/`、对象文件与常见临时文件。
- Windows 上出现换行符提示（LF→CRLF）？
  - 可设置：`git config core.autocrlf true`。
- SSH 推送被 22 端口拦截？
  - 使用 HTTPS，或把远程设置为 `ssh://git@ssh.github.com:443/<owner>/<repo>.git`。

---

## 贡献
欢迎提交 Issue 与 Pull Request：
1. Fork 本仓库并创建分支
2. 完成修改并通过 `ctest`
3. 提交 PR 说明改动与动机

---

## 许可证
尚未设置。如需开源，请在根目录添加 `LICENSE`（例如 MIT/Apache-2.0）。

---

## 致谢
本项目部分模块与测试来自课程/实验场景，感谢相关资料与社区贡献者。
