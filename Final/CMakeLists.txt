cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( Final )
find_package( OpenCV REQUIRED )
add_executable( Final final.cpp )
target_link_libraries(Final ${OpenCV_LIBS} )
