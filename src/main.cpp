// This file is based on tutorial that can be found:
// "Harnessing the POWER of Your Graphics Card💪 | An Introduction to OpenCL"
// source: https://www.youtube.com/watch?v=Iz6feoh9We8&t=0s
// author: NoNumberMan
// accessed 27th April 2025
// Code provided in the tutorial was adjusted to project needs.


#include <CL/cl.h>
#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define TRACE(fmt, ...)                                                            \
    {                                                                           \
        printf("[TRACE_FMT] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    };

std::string loadKernelSource(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + filePath);
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

const std::string REPOROOT = std::filesystem::current_path().parent_path().string();
const std::string KERNELS = REPOROOT + "/kernels/";
const std::string KERNEL_FILE = KERNELS + "kernels.cl";
const std::string IMAGES = REPOROOT + "/images/";

int main(int argc, char* args[]) {

    std::cout << cv::getBuildInformation() << std::endl;

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
    cl_command_queue_properties properties = 0;
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, device, &properties, &commandQueueResult);
    assert(commandQueueResult == CL_SUCCESS);
    TRACE("Command queue created successfully");

    TRACE("Loading kernel source");
    std::string kernelSource = loadKernelSource(KERNEL_FILE);
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

    cv::Mat source_image = cv::imread(IMAGES + "sample.jpg");
    if (source_image.empty()) {
        std::cerr << "Failed to load image\n";
        return -1;
    }
    cv::Mat sourceRGBA;
    cv::cvtColor(source_image, sourceRGBA, cv::COLOR_BGR2RGBA);

    cl_image_desc desc = {};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = sourceRGBA.cols;
    desc.image_height = sourceRGBA.rows;
    desc.image_depth = 0;
    desc.image_array_size = 1;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = nullptr;

    cl_image_format format = {};
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT8;
    
    cl_mem image_in = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    &format, &desc, sourceRGBA.data, &contextResult);
    assert(contextResult == CL_SUCCESS);
    cl_mem image_out = clCreateImage(context, CL_MEM_READ_WRITE,
                                     &format, &desc, nullptr, &contextResult);
    assert(contextResult == CL_SUCCESS);
    
    cl_int kernelResult;
    cl_kernel black_kernel = clCreateKernel(program, "make_black_image", &kernelResult);
    assert(kernelResult == CL_SUCCESS);
    cl_kernel increment_kernel = clCreateKernel(program, "increment_pixel", &kernelResult);
    assert(kernelResult == CL_SUCCESS);

    cl_int argResult;
    argResult = clSetKernelArg(black_kernel, 0, sizeof(cl_mem), &image_in);
    assert(argResult == CL_SUCCESS);
    argResult = clSetKernelArg(black_kernel, 1, sizeof(cl_mem), &image_out);
    assert(argResult == CL_SUCCESS);
    argResult = clSetKernelArg(increment_kernel, 0, sizeof(cl_mem), &image_out);
    assert(argResult == CL_SUCCESS);

    size_t globalWorkSize[2] = {static_cast<size_t>(sourceRGBA.cols), static_cast<size_t>(sourceRGBA.rows)};
    cl_int enqueueResult = clEnqueueNDRangeKernel(commandQueue, black_kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    assert(enqueueResult == CL_SUCCESS);

    for (int i = 0; i < 50; ++i)
    {
        enqueueResult = clEnqueueNDRangeKernel(commandQueue, increment_kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
        assert(enqueueResult == CL_SUCCESS);
    }
    clFinish(commandQueue);

    TRACE("Reading back image");
    cv::Mat outputRGBA(sourceRGBA.size(), sourceRGBA.type());
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {static_cast<size_t>(sourceRGBA.cols), static_cast<size_t>(sourceRGBA.rows), 1};
    cl_int readResult = clEnqueueReadImage(commandQueue, image_out, CL_TRUE, origin, region, 0, 0, outputRGBA.data, 0, nullptr, nullptr);
    assert(readResult == CL_SUCCESS);
    TRACE("Image read back successfully");
    TRACE("Writing output image");
    cv::imwrite(IMAGES + "output.jpg", outputRGBA);
    TRACE("Output image written successfully");

    TRACE("Releasing resources");
    clReleaseMemObject(image_in);
    clReleaseMemObject(image_out);
    clReleaseKernel(black_kernel);
    clReleaseKernel(increment_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    TRACE("Resources released successfully");

    TRACE("PR25LAAW05_SUPERPIXEL application finished");

    return 0;
}
