cmake_minimum_required(VERSION 3.10)
project(PR25Laaw05_SUPERPIXEL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/inc)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(PR25Laaw05_SUPERPIXEL ${SOURCES})

# Link both OpenCV and OpenCL
target_link_libraries(PR25Laaw05_SUPERPIXEL ${OpenCV_LIBS} OpenCL::OpenCL)

# Add OpenCV include directories
target_include_directories(PR25Laaw05_SUPERPIXEL PRIVATE ${OpenCV_INCLUDE_DIRS})
