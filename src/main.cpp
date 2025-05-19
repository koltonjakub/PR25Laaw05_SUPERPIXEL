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
#include <map>

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

const std::string REPOROOT = std::filesystem::current_path().parent_path().string()+"/PR25Laaw05_SUPERPIXEL";
const std::string KERNELS = REPOROOT + "/kernels/";
const std::string KERNEL_FILE = KERNELS + "kernels.cl";
const std::string IMAGES = REPOROOT + "/images/";
const std::map<std::string, std::string> IMAGES_MAP = {
    {"3",  IMAGES + "(BARCODE)0003.tif"},
    {"7",  IMAGES + "(BARCODE)0007.tif"},
    {"15", IMAGES + "(BARCODE)0015.tif"},
    {"5",  IMAGES + "BG_0005.tif"},
    {"12", IMAGES + "BG_0012.tif"},
    {"14", IMAGES + "BG_0014.tif"},
};

int main(int argc, char* args[]) {

    //std::cout << cv::getBuildInformation() << std::endl;

    TRACE("PR25LAAW05_SUPERPIXEL application started");

    TRACE("Getting platform ids");
    cl_platform_id platforms[64];
    unsigned int platformCount;
    cl_int platformResult = clGetPlatformIDs(64, platforms, &platformCount);
    assert(platformResult == CL_SUCCESS);
    TRACE("Number of platforms: %u", platformCount);

    TRACE("Getting device info");
    // cl_device_id device = nullptr;
    // for(int i = 0; i < platformCount && device == nullptr; ++i)
    // {
    //     cl_device_id devices[64];
    //     unsigned int deviceCount;
    //     cl_int deviceResult = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);

    //     if(deviceResult == CL_SUCCESS)
    //     {
    //         TRACE("deviceResult SUCCESS. Number of devices: %u", deviceCount);
    //         for(int j = 0; j < deviceCount; j++)
    //         {
    //             char vendorName[256];
    //             size_t vendorNameLength;
    //             cl_int deviceInfoResult = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 256, vendorName, &vendorNameLength);
    //             TRACE("Found device vendor: %s", vendorName);
    //             if(deviceInfoResult == CL_SUCCESS && std::string(vendorName).substr(0, vendorNameLength) == "NVIDIA Corporation")
    //             {
    //                 device = devices[j];
    //                 TRACE("Choosing dedicated NVIDIA GPU found");
    //                 break;
    //             }
    //         }
    //     }
    // }

    // New plattform selection code
    TRACE("Available OpenCL platforms:");
    for (unsigned int i = 0; i < platformCount; ++i) {
        char name[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        TRACE("Platform %u: %s", i, name);
    }


    cl_device_id device = nullptr;
    int platformIndex = 0; // default
    if (argc > 2) {
        platformIndex = std::stoi(args[2]);
    }
    if (platformIndex >= static_cast<int>(platformCount)) {
        std::cerr << "Invalid platform index.\n";
        return -1;
    }
    
    cl_device_id devices[64];
    unsigned int deviceCount;
    cl_int deviceResult = clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_ALL, 64, devices, &deviceCount);
    if (deviceResult != CL_SUCCESS || deviceCount == 0) {
        std::cerr << "No GPU devices found on selected platform.\n";
        return -1;
    }
    device = devices[0]; // Use first device by default
    TRACE("Selected platform index %d", platformIndex);

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

    TRACE("Loading image %s", IMAGES_MAP.at(args[1]).c_str());
    cv::Mat source_image = cv::imread(IMAGES_MAP.at(args[1]).c_str());
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
    cl_mem mask_image = clCreateImage(context, CL_MEM_READ_WRITE,
                                     &format, &desc, nullptr, &contextResult);
    assert(contextResult == CL_SUCCESS);
    cl_mem hsv_image = clCreateImage(context, CL_MEM_READ_WRITE,
                                    &format, &desc, nullptr, &contextResult);
    assert(contextResult == CL_SUCCESS);

    cl_int kernelResult;
    cl_kernel hsv_binary_kernel = clCreateKernel(program, "hsv_binary_filter", &kernelResult);
    assert(kernelResult == CL_SUCCESS);

    cl_int argResult;
    argResult = clSetKernelArg(hsv_binary_kernel, 0, sizeof(cl_mem), &image_in);
    assert(argResult == CL_SUCCESS);
    argResult = clSetKernelArg(hsv_binary_kernel, 1, sizeof(cl_mem), &mask_image);
    assert(argResult == CL_SUCCESS);
    argResult = clSetKernelArg(hsv_binary_kernel, 2, sizeof(cl_mem), &hsv_image);
    assert(argResult == CL_SUCCESS);

    size_t globalWorkSize[2] = {static_cast<size_t>(sourceRGBA.cols), static_cast<size_t>(sourceRGBA.rows)};
    cl_int enqueueResult = clEnqueueNDRangeKernel(commandQueue, hsv_binary_kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    clFinish(commandQueue);

    TRACE("Reading back image");
    cv::Mat outputRGBA(sourceRGBA.size(), sourceRGBA.type());
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {static_cast<size_t>(sourceRGBA.cols), static_cast<size_t>(sourceRGBA.rows), 1};
    cl_int readResult = clEnqueueReadImage(commandQueue, mask_image, CL_TRUE, origin, region, 0, 0, outputRGBA.data, 0, nullptr, nullptr);
    assert(readResult == CL_SUCCESS);
    TRACE("Image read back successfully");
    TRACE("Writing mask_image image");
    cv::imwrite(IMAGES + "mask_image.jpg", outputRGBA);
    TRACE("mask_image written successfully");

    TRACE("Reading back image");
    readResult = clEnqueueReadImage(commandQueue, hsv_image, CL_TRUE, origin, region, 0, 0, outputRGBA.data, 0, nullptr, nullptr);
    assert(readResult == CL_SUCCESS);
    TRACE("Image read back successfully");
    TRACE("Writing hsv_image image");
    cv::imwrite(IMAGES + "hsv_image.jpg", outputRGBA);
    TRACE("hsv_image written successfully");

    // ---- ADDING assignPixelsToClusters ----

    TRACE("Creating assignPixelsToClusters kernel");
    cl_kernel cluster_kernel = clCreateKernel(program, "assignPixelsToClusters", &kernelResult);
    assert(kernelResult == CL_SUCCESS);

    const int width = sourceRGBA.cols;
    const int height = sourceRGBA.rows;
    const int numClusters = 100; // Tune this to your needs
    const float m = 10.0f;       // Compactness factor for SLIC
    const int step = static_cast<int>(sqrt((width * height) / numClusters));
    int clustersX = width / step;
    int clustersY = height / step;

    std::vector<float> clusterData(numClusters * 5);

    int c = 0;
    for (int y = step / 2; y < height && c < numClusters; y += step) {
        for (int x = step / 2; x < width && c < numClusters; x += step) {
            clusterData[c * 5 + 0] = 0.5f;         // H
            clusterData[c * 5 + 1] = 0.5f;         // S
            clusterData[c * 5 + 2] = 0.5f;         // V
            clusterData[c * 5 + 3] = (float)x;     // x
            clusterData[c * 5 + 4] = (float)y;     // y
            ++c;
        }
    }

    // Buffers for clusters, labels, and distances
    cl_mem clusterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * clusterData.size(), clusterData.data(), &contextResult);
    assert(contextResult == CL_SUCCESS);

    cl_mem labelBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        sizeof(int) * width * height, nullptr, &contextResult);
    assert(contextResult == CL_SUCCESS);

    cl_mem distanceBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                           sizeof(float) * width * height, nullptr, &contextResult);
    assert(contextResult == CL_SUCCESS);

    // Set kernel args
    clSetKernelArg(cluster_kernel, 0, sizeof(cl_mem), &hsv_image);
    clSetKernelArg(cluster_kernel, 1, sizeof(int), &width);
    clSetKernelArg(cluster_kernel, 2, sizeof(int), &height);
    clSetKernelArg(cluster_kernel, 3, sizeof(cl_mem), &clusterBuffer);
    clSetKernelArg(cluster_kernel, 4, sizeof(int), &numClusters);
    clSetKernelArg(cluster_kernel, 5, sizeof(float), &m);
    clSetKernelArg(cluster_kernel, 6, sizeof(cl_mem), &labelBuffer);
    clSetKernelArg(cluster_kernel, 7, sizeof(cl_mem), &distanceBuffer);

    TRACE("Launching assignPixelsToClusters kernel");
    clEnqueueNDRangeKernel(commandQueue, cluster_kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    clFinish(commandQueue);
    TRACE("assignPixelsToClusters kernel finished");

    // (Optional) Read back labels to check or visualize
    std::vector<int> labels(width * height);
    clEnqueueReadBuffer(commandQueue, labelBuffer, CL_TRUE, 0, sizeof(int) * labels.size(), labels.data(), 0, nullptr, nullptr);

    // Save label map as grayscale image (basic visualization)
    cv::Mat labelImg(height, width, CV_8UC1);
    for (int i = 0; i < labels.size(); ++i) {
        labelImg.data[i] = static_cast<uchar>((labels[i] * 17) % 255); // crude mapping
    }
    cv::imwrite(IMAGES + "label_image.jpg", labelImg);
    TRACE("label_image written successfully");

    // Release new resources
    TRACE("Releasing resources");
    clReleaseMemObject(clusterBuffer);
    clReleaseMemObject(labelBuffer);
    clReleaseMemObject(distanceBuffer);
    clReleaseKernel(cluster_kernel);
    clReleaseMemObject(image_in);
    clReleaseMemObject(mask_image);
    clReleaseMemObject(hsv_image);
    clReleaseKernel(hsv_binary_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    TRACE("Resources released successfully");

    TRACE("PR25LAAW05_SUPERPIXEL application finished");

    return 0;
}
