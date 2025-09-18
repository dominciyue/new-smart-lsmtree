cmake_minimum_required(VERSION 3.10)
project(LSM_TREE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find threads package for ThreadPool
find_package(Threads REQUIRED)

if (MINGW)
    add_compile_definitions(_WIN32_WINNT=0x0A00)  # Windows 10
    add_compile_definitions(NOMINMAX)              # 避免 Windows.h 中的 min/max 宏冲突
endif()

# 添加llama.cpp的包含路径
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third_party/llama.cpp)
include_directories(${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)
include_directories(${CMAKE_SOURCE_DIR}/embedding)

# 源文件列表
set(SOURCE_FILES
        kvstore.cc
        bloom.cpp
        sstable.cpp
        sstablehead.cpp
        skiplist.cpp
)

# 头文件列表
set(HEADER_FILES
        kvstore.h
        kvstore_api.h
        skiplist.h
        bloom.h
        sstable.h
        sstablehead.h
        MurmurHash3.h
        utils.h
        test.h
)

# correctness测试可执行文件
add_executable(correctness
        correctness.cc
        # SOURCE_FILES and HEADER_FILES are implicitly part of kvstore library
)
# Threads::Threads is linked to kvstore, which correctness links to.
# If correctness.cc itself uses threads directly, link it here too.
target_link_libraries(correctness PUBLIC kvstore) # kvstore already links embedding

# persistence测试可执行文件
add_executable(persistence
        persistence.cc
)
target_link_libraries(persistence PUBLIC kvstore)

# Phase 5 large data test executable
add_executable(phase5_test
        phase5_large_data_test.cc
)
# phase5_test links kvstore (which includes thread logic)
# and directly uses std::filesystem (usually covered by C++17/20)
# If Threads::Threads is not transitively linked via kvstore's PUBLIC interface for ThreadPool,
# link it directly: target_link_libraries(phase5_test PRIVATE Threads::Threads)
target_link_libraries(phase5_test PUBLIC kvstore)


add_library(kvstore STATIC
        ${SOURCE_FILES}
        # HEADER_FILES are for IDEs/source groups, not strictly for library compilation here
        # unless they are also source files (e.g. template implementations .cpp/.tpp)
)
# Link kvstore with embedding and Threads (for ThreadPool within kvstore.cc)
target_link_libraries(kvstore PUBLIC embedding Threads::Threads)

# 添加 llama.cpp 子目录
add_subdirectory(third_party/llama.cpp)

# 添加 embedding 子目录
add_subdirectory(embedding)

# 添加 test 子目录
add_subdirectory(test)
