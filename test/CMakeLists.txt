cmake_minimum_required(VERSION 3.10)

add_executable(test-fallback-simple test_fallback_simple.c)

target_link_libraries(test-fallback-simple 
    PRIVATE 
    ggml-cpu
    ggml
)

target_include_directories(test-fallback-simple 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
)
