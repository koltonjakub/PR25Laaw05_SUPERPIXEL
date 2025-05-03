// This file is based on tutorial that can be found:
// "Harnessing the POWER of Your Graphics Card💪 | An Introduction to OpenCL"
// source: https://www.youtube.com/watch?v=Iz6feoh9We8&t=0s
// author: NoNumberMan
// accessed 27th April 2025


#include <CL/cl.h>
#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define TRACE(fmt, ...)                                                            \
    {                                                                           \
        printf("[TRACE_FMT] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    };

// Function to load kernel source from a file
std::string loadKernelSource(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + filePath);
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

int main(int argc, char* args[]) {
    TRACE("PR25LAAW05_SUPERPIXEL application started");

    TRACE("Getting platform ids");
    cl_platform_id platforms[64];
    unsigned int platformCount;
    cl_int platformResult = clGetPlatformIDs(64, platforms, &platformCount);
    assert(platformResult == CL_SUCCESS);
    TRACE("Number of platforms: %u", platformCount);

    TRACE("Getting device info");
    cl_device_id device = nullptr;
    for(int i = 0; i < platformCount && device == nullptr; ++i)
    {
        cl_device_id devices[64];
        unsigned int deviceCount;
        cl_int deviceResult = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);

        if(deviceResult == CL_SUCCESS)
        {
            TRACE("deviceResult SUCCESS. Number of devices: %u", deviceCount);
            for(int j = 0; j < deviceCount; j++)
            {
                char vendorName[256];
                size_t vendorNameLength;
                cl_int deviceInfoResult = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 256, vendorName, &vendorNameLength);
                TRACE("Found device vendor: %s", vendorName);
                if(deviceInfoResult == CL_SUCCESS && std::string(vendorName).substr(0, vendorNameLength) == "NVIDIA Corporation")
                {
                    device = devices[j];
                    TRACE("Choosing dedicated NVIDIA GPU found");
                    break;
                }
            }
        }
    }

    TRACE("Creating context");
    cl_int contextResult;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &contextResult);
    assert(contextResult == CL_SUCCESS);
    TRACE("Context created successfully");

    TRACE("Creating command queue");
    cl_int commandQueueResult;
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &commandQueueResult);
    assert(commandQueueResult == CL_SUCCESS);
    TRACE("Command queue created successfully");

    TRACE("Loading kernel source");
    std::string kernelSource = loadKernelSource("../kernels/vector_sum.cl");
    const char* programSource = kernelSource.c_str();
    size_t programSourceLength = kernelSource.size();

    TRACE("Creating program");
    cl_int programResult;
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, &programSourceLength, &programResult);
    assert(programResult == CL_SUCCESS);
    TRACE("Program created successfully");

    TRACE("Building program");
    cl_int programBuildResult = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (programBuildResult != CL_SUCCESS) {
        size_t logLength;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLength);
        std::vector<char> log(logLength);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logLength, log.data(), nullptr);
        TRACE("Program build failed:\n%s", log.data());
        return -1;
    }
    TRACE("Program built successfully");

    TRACE("Creating buffers");
    cl_int bufferResult;

    // Input data
    float vector_a_data[2] = {1.0f, 2.0f};
    float vector_b_data[2] = {3.0f, 4.0f};

    // Create input buffers
    cl_mem vector_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 2, vector_a_data, &bufferResult);
    assert(bufferResult == CL_SUCCESS);
    cl_mem vector_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 2, vector_b_data, &bufferResult);
    assert(bufferResult == CL_SUCCESS);

    // Create output buffer
    cl_mem vector_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * 2, nullptr, &bufferResult);
    assert(bufferResult == CL_SUCCESS);
    TRACE("Buffers created successfully");

    TRACE("Setting kernel arguments");
    cl_int kernelResult;
    cl_kernel kernel = clCreateKernel(program, "vector_sum", &kernelResult);
    assert(kernelResult == CL_SUCCESS);

    cl_int argResult = clSetKernelArg(kernel, 0, sizeof(cl_mem), &vector_a);
    assert(argResult == CL_SUCCESS);
    argResult = clSetKernelArg(kernel, 1, sizeof(cl_mem), &vector_b);
    assert(argResult == CL_SUCCESS);
    argResult = clSetKernelArg(kernel, 2, sizeof(cl_mem), &vector_c);
    assert(argResult == CL_SUCCESS);
    const int count{2};
    argResult = clSetKernelArg(kernel, 3, sizeof(int), &count);
    assert(argResult == CL_SUCCESS);
    TRACE("Kernel arguments set successfully");

    TRACE("Enqueuing kernel");
    size_t globalWorkSize[1] = {2}; // Process 2 elements
    cl_int enqueueResult = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    assert(enqueueResult == CL_SUCCESS);
    TRACE("Kernel enqueued successfully");

    TRACE("Reading back results");
    float vector_c_data[2];
    cl_int readResult = clEnqueueReadBuffer(commandQueue, vector_c, CL_TRUE, 0, sizeof(float) * 2, vector_c_data, 0, nullptr, nullptr);
    assert(readResult == CL_SUCCESS);
    TRACE("Results read successfully");

    TRACE("Output:");
    for (int i = 0; i < 2; ++i) {
        printf("vector_c[%d] = %f\n", i, vector_c_data[i]);
    }

    // Cleanup
    clReleaseMemObject(vector_a);
    clReleaseMemObject(vector_b);
    clReleaseMemObject(vector_c);
    clReleaseKernel(kernel);
}
