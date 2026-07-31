cmake_minimum_required(VERSION 3.16)
project(nchw_pooling_examples LANGUAGES CXX)

find_package(Threads REQUIRED)

add_library(nchw_pooling_examples OBJECT
    ref_pooling.cpp
    jit_pooling.cpp)

target_compile_features(nchw_pooling_examples PRIVATE cxx_std_17)
target_link_libraries(nchw_pooling_examples PRIVATE Threads::Threads)

add_executable(nchw_pooling_test
    pooling_test.cpp
    ref_pooling.cpp
    jit_pooling.cpp)

target_compile_features(nchw_pooling_test PRIVATE cxx_std_17)
target_link_libraries(nchw_pooling_test PRIVATE Threads::Threads)
