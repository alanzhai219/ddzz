cmake_minimum_required(VERSION 3.14)
project(lru_cache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only library
add_library(lru_cache INTERFACE)
target_include_directories(lru_cache INTERFACE include)

# Tests
enable_testing()
add_executable(test_cache test/test_cache.cpp)
target_link_libraries(test_cache PRIVATE lru_cache)
add_test(NAME test_cache COMMAND test_cache)
