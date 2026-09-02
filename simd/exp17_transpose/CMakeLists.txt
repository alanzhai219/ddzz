cmake_minimum_required(VERSION 3.16)
project(exp17_transpose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

add_executable(exp17_transpose
    main.cpp
    fp32_transpose.cpp)

target_compile_options(exp17_transpose PRIVATE -O3 -Wall -Wextra -Wpedantic)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(exp17_transpose PRIVATE -mavx2)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(exp17_transpose PRIVATE OpenMP::OpenMP_CXX)
endif()
