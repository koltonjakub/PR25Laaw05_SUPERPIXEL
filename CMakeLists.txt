cmake_minimum_required(VERSION 3.10)
project(OpenCLExample)

# C++ settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export REPOROOT as a constant variable pointing to the root of the project
set(REPOROOT ${PROJECT_SOURCE_DIR})

# Include and source directories
include_directories(${PROJECT_SOURCE_DIR}/inc)
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Add executable and link
add_executable(OpenCLExample ${SOURCES})
target_link_libraries(OpenCLExample OpenCL::OpenCL)
