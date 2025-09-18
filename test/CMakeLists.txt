add_executable(Embedding_Test Embedding_Test.cpp)

target_link_libraries(Embedding_Test PUBLIC embedding)

add_executable(E2E_Test E2E_test.cpp)

target_link_libraries(E2E_Test PUBLIC kvstore embedding)

# --- 新增 Phase 4 测试目标 ---
add_executable(Vector_Persistent_Test_Phase1 ${CMAKE_SOURCE_DIR}/Vector_Persistent_Test_Phase1.cpp)
target_link_libraries(Vector_Persistent_Test_Phase1 PUBLIC kvstore embedding)

add_executable(Vector_Persistent_Test_Phase2 ${CMAKE_SOURCE_DIR}/Vector_Persistent_Test_Phase2.cpp)
target_link_libraries(Vector_Persistent_Test_Phase2 PUBLIC kvstore embedding)

add_executable(HNSW_Delete_Test ${CMAKE_SOURCE_DIR}/HNSW_Delete_Test.cpp)
target_link_libraries(HNSW_Delete_Test PUBLIC kvstore embedding)

add_executable(HNSW_Persistent_Test_Phase1 ${CMAKE_SOURCE_DIR}/HNSW_Persistent_Test_Phase1.cpp)
target_link_libraries(HNSW_Persistent_Test_Phase1 PUBLIC kvstore embedding)

add_executable(HNSW_Persistent_Test_Phase2 ${CMAKE_CURRENT_SOURCE_DIR}/../HNSW_Persistent_Test_Phase2.cpp)
target_link_libraries(HNSW_Persistent_Test_Phase2 PUBLIC kvstore common llama embedding ggml)

# New test for 100k data
# add_executable(LargeScale_Persistence_Test LargeScale_Persistence_Test.cpp)
# target_link_libraries(LargeScale_Persistence_Test PUBLIC kvstore common llama embedding ggml)

# Optional: Add specific compile definitions if needed for this test, e.g., to point to data files if not relative
# target_compile_definitions(LargeScale_Persistence_Test PRIVATE LARGE_SCALE_TEXT_FILE_PATH="path/to/your/cleaned_text_100k.txt")

# Ensure all tests are part of the `test` target or an overall `check` target if you have one
