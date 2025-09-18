cmake_minimum_required(VERSION 3.14)

add_executable(bench-prefetch-k bench-prefetch-k.cpp)

target_link_libraries(bench-prefetch-k PRIVATE llama ggml Threads::Threads)

target_include_directories(bench-prefetch-k PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

