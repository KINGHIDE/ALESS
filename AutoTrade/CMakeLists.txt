cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(AutoTrade)

# LibTorchのパスを設定
set(CMAKE_PREFIX_PATH "/home/hide/libtorch")

find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
add_executable(AutoTrade main.cpp)
target_link_libraries(AutoTrade ${TORCH_LIBRARIES} ${CURL_LIBRARIES})
target_include_directories(AutoTrade PRIVATE ${CURL_INCLUDE_DIRS})
set_property(TARGET AutoTrade PROPERTY CXX_STANDARD 14)
