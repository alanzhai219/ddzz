cmake_minimum_required(VERSION 3.16)
project(gpt2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O2 -Wall -Wextra)

add_executable(gpt2
    main.cpp
    tensor.cpp
    ops.cpp
    tokenizer.cpp
    weights.cpp
    modeling.cpp
)
