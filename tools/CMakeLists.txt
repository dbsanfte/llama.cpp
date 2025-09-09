# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(main)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
    add_subdirectory(run)
    add_subdirectory(tokenize)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL)
        # these examples use the backends directly and cannot be built with dynamic loading
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    
    # GGML tensor visualization tools for debugging
    add_executable(visualize-tensor-basic visualize-tensor-basic.cpp)
    target_link_libraries(visualize-tensor-basic PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(visualize-tensor-basic PRIVATE cxx_std_11)
    
    add_executable(visualize-tensor-quantization visualize-tensor-quantization.cpp)
    target_link_libraries(visualize-tensor-quantization PRIVATE ggml ggml-cpu ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(visualize-tensor-quantization PRIVATE cxx_std_11)
    
    add_executable(visualize-tensor-add-types visualize-tensor-add-types.cpp)
    target_link_libraries(visualize-tensor-add-types PRIVATE ggml ggml-cpu ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(visualize-tensor-add-types PRIVATE cxx_std_11)
    
    add_executable(visualize-tensor-broadcasting visualize-tensor-broadcasting.cpp)
    target_link_libraries(visualize-tensor-broadcasting PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(visualize-tensor-broadcasting PRIVATE cxx_std_11)
endif()
