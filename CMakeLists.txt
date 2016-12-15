
cmake_minimum_required(VERSION 2.7)
project(ImageCollector)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#Opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#Caffe
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

set(SOURCE_FILES imagecollector.cpp)
add_executable(image_collector ${SOURCE_FILES} )
target_link_libraries(image_collector ${Caffe_LIBRARIES} ${OpenCV_LIBS})