project(MiniDNN)
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

add_library(MiniDNN INTERFACE)
target_include_directories(MiniDNN INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(eigen3 INTERFACE)
target_include_directories(eigen3 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)

add_executable(MiniDNN_test1 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/test1/test1.cpp)
target_link_libraries(MiniDNN_test1 MiniDNN eigen3)

add_executable(MiniDNN_test2 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/test2/test2.cpp)
target_link_libraries(MiniDNN_test2 MiniDNN eigen3)
