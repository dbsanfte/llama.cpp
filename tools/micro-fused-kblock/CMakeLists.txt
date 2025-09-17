set(TARGET micro-fused-kblock)
add_executable(${TARGET} micro-fused-kblock.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
# Allow profiling env usage without extra flags
