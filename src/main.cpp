#include <CL/cl.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define TRACE(fmt, ...)                                                            \
    { printf("[TRACE_FMT] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); };

const std::string REPOROOT = std::filesystem::current_path().parent_path().string() + "/PR25Laaw05_SUPERPIXEL";
const std::string KERNELS = REPOROOT + "/kernels/";
const std::string KERNEL_FILE = KERNELS + "kernels.cl";
const std::string IMAGES = REPOROOT + "/images/";
const std::map<std::string, std::string> IMAGES_MAP = {
    {"0", IMAGES + "sample.png"},
    {"05", IMAGES + "(BARCODE)0005.tif"},
    {"3",  IMAGES + "(BARCODE)0003.tif"},
    {"7",  IMAGES + "(BARCODE)0007.tif"},
    {"15", IMAGES + "(BARCODE)0015.tif"},
    {"5",  IMAGES + "BG_0005.tif"},
    {"12", IMAGES + "BG_0012.tif"},
    {"14", IMAGES + "BG_0014.tif"},
};

std::string loadKernelSource(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) throw std::runtime_error("Failed to open kernel file: " + filePath);
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

cl_device_id selectDevice(int platformIndex, cl_platform_id* platforms, unsigned int platformCount) {
    if (platformIndex >= static_cast<int>(platformCount)) throw std::runtime_error("Invalid platform index.");
    cl_device_id devices[64];
    unsigned int deviceCount;
    cl_int result = clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_ALL, 64, devices, &deviceCount);
    if (result != CL_SUCCESS || deviceCount == 0) throw std::runtime_error("No OpenCL devices found.");
    return devices[0]; // Default: first device
}

cv::Mat loadAndConvertImage(const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) throw std::runtime_error("Failed to load image.");
    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
    return rgba;
}

cl_mem createImage(cl_context context, cl_mem_flags flags, cl_image_format format,
                   cl_image_desc desc, void* hostPtr = nullptr) {
    cl_int result;
    cl_mem image = clCreateImage(context, flags, &format, &desc, hostPtr, &result);
    assert(result == CL_SUCCESS);
    return image;
}

cl_program buildProgram(cl_context context, cl_device_id device, const std::string& source) {
    const char* src = source.c_str();
    size_t len = source.length();
    cl_int result;
    cl_program program = clCreateProgramWithSource(context, 1, &src, &len, &result);
    assert(result == CL_SUCCESS);
    result = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (result != CL_SUCCESS) {
        size_t logLength;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLength);
        std::vector<char> log(logLength);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logLength, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        exit(-1);
    }
    return program;
}

void writeImage(const std::string& path, cl_command_queue queue, cl_mem image,
                int width, int height, int type = CV_8UC4) {
    cv::Mat output(height, width, type);
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {static_cast<size_t>(width), static_cast<size_t>(height), 1};
    cl_int res = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0, output.data, 0, nullptr, nullptr);
    assert(res == CL_SUCCESS);
    cv::imwrite(path, output);
}


void createInitialClusters(int width, int height, int& numClusters, std::vector<float>& clusterData) {
    // Estimate grid dimensions based on aspect ratio and target cluster count
    int gridCols = static_cast<int>(std::sqrt((float)numClusters * width / height));
    int gridRows = static_cast<int>(std::ceil((float)numClusters / gridCols));

    // Override number of clusters to fit full grid
    numClusters = gridCols * gridRows;
    clusterData.resize(numClusters * 5);  // H, S, V, x, y

    float stepX = static_cast<float>(width) / gridCols;
    float stepY = static_cast<float>(height) / gridRows;

    int c = 0;
    for (int row = 0; row < gridRows; ++row) {
        for (int col = 0; col < gridCols; ++col) {
            float cx = (col + 0.5f) * stepX;
            float cy = (row + 0.5f) * stepY;

            // Clamp to image bounds
            if (cx >= width) cx = width - 1;
            if (cy >= height) cy = height - 1;

            clusterData[c * 5 + 0] = 0.5f;      // H
            clusterData[c * 5 + 1] = 0.5f;      // S
            clusterData[c * 5 + 2] = 0.5f;      // V
            clusterData[c * 5 + 3] = cx;        // X
            clusterData[c * 5 + 4] = cy;        // Y

            ++c;
        }
    }

    std::cout << "Adjusted numClusters = " << numClusters << " (" << gridCols << " × " << gridRows << " grid)" << std::endl;
}


const int NUM_ITERATIONS = 10;

void assertCLSuccess(cl_int result, const char* message) {
    if (result != CL_SUCCESS) {
        std::cerr << message << " Error Code: " << result << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

std::tuple<cl_context, cl_command_queue> createOpenCLContextAndQueue(cl_device_id device) {
    cl_int result;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &result);
    assertCLSuccess(result, "Failed to create OpenCL context");

    cl_command_queue_properties props[] = {0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &result);
    assertCLSuccess(result, "Failed to create command queue");

    return {context, queue};
}

cv::Mat loadAndConvertImage(const std::string& path, cl_context context, cl_mem& imageCL, cl_image_format format, cl_image_desc desc) {
    TRACE("Loading image %s", path.c_str());
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Image load failed: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cv::Mat imgRGBA;
    cv::cvtColor(img, imgRGBA, cv::COLOR_BGR2RGBA);

    cl_int result;
    imageCL = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &desc, imgRGBA.data, &result);
    assertCLSuccess(result, "Failed to create OpenCL image");

    return imgRGBA;
}

void runAssignPixelsToClusters(cl_command_queue queue, cl_kernel kernel, cl_mem hsv_image, int width, int height,
                                cl_mem clusterBuffer, int numClusters, float m, cl_mem labelBuffer, cl_mem distanceBuffer) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &hsv_image);
    clSetKernelArg(kernel, 1, sizeof(int), &width);
    clSetKernelArg(kernel, 2, sizeof(int), &height);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &clusterBuffer);
    clSetKernelArg(kernel, 4, sizeof(int), &numClusters);
    clSetKernelArg(kernel, 5, sizeof(float), &m);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &labelBuffer);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &distanceBuffer);

    size_t globalWorkSize[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
}

void runUpdateClusters(cl_command_queue queue, cl_kernel updateKernel, cl_mem hsv_image, cl_mem labelBuffer,
                       int width, int height, int numClusters, cl_mem clusterSumBuffer, cl_mem clusterCountBuffer) {
    clSetKernelArg(updateKernel, 0, sizeof(cl_mem), &hsv_image);
    clSetKernelArg(updateKernel, 1, sizeof(cl_mem), &labelBuffer);
    clSetKernelArg(updateKernel, 2, sizeof(int), &width);
    clSetKernelArg(updateKernel, 3, sizeof(int), &height);
    clSetKernelArg(updateKernel, 4, sizeof(int), &numClusters);
    clSetKernelArg(updateKernel, 5, sizeof(cl_mem), &clusterSumBuffer);
    clSetKernelArg(updateKernel, 6, sizeof(cl_mem), &clusterCountBuffer);

    size_t globalWorkSize[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    clEnqueueNDRangeKernel(queue, updateKernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
}

void visualizeLabels(const std::string& path, const std::vector<int>& labels,
                     int width, int height, int numClusters) {
    cv::Mat image(height, width, CV_8UC3);
    std::vector<cv::Vec3b> colors(numClusters);

    // Assign random colors to each cluster
    cv::RNG rng(12345);
    for (int i = 0; i < numClusters; ++i) {
        colors[i] = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int label = labels[y * width + x];
            if (label >= 0 && label < numClusters)
                image.at<cv::Vec3b>(y, x) = colors[label];
            else
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fallback for invalid label
        }
    }

    cv::imwrite(path, image);
}

void visualizeLabelBoundaries(const std::string& inputImagePath,
                              const std::string& outputImagePath,
                              const std::vector<int>& labels,
                              int width, int height, int numClusters) {
    // Load the original image
    cv::Mat image = cv::imread(inputImagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load input image: " << inputImagePath << std::endl;
        return;
    }

    // Ensure image dimensions match label data
    if (image.cols != width || image.rows != height) {
        std::cerr << "Error: Image size does not match given width and height." << std::endl;
        return;
    }

    auto isBoundary = [&](int x, int y) {
        int currentLabel = labels[y * width + x];
        // Check 4-neighborhood
        if (x > 0 && labels[y * width + (x - 1)] != currentLabel) return true;
        if (x < width - 1 && labels[y * width + (x + 1)] != currentLabel) return true;
        if (y > 0 && labels[(y - 1) * width + x] != currentLabel) return true;
        if (y < height - 1 && labels[(y + 1) * width + x] != currentLabel) return true;
        return false;
    };

    // Draw boundary pixels on the image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (isBoundary(x, y)) {
                if((x+y)%2){
                    image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // red boundary
                }
                else
                {
                    image.at<cv::Vec3b>(y, x) = cv::Vec3b(100, 100, 255); // boundary
                }
                
                    
            }
        }
    }

    // Save the output image
    cv::imwrite(outputImagePath, image);
}


int main(int argc, char* args[]) {
    TRACE("PR25LAAW05_SUPERPIXEL application started");

    // OpenCL Setup
    cl_platform_id platforms[64];
    unsigned int platformCount;
    clGetPlatformIDs(64, platforms, &platformCount);

    for (unsigned int i = 0; i < platformCount; ++i) {
        char name[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        TRACE("Platform %u: %s", i, name);
    }

    int platformIndex = (argc > 2) ? std::stoi(args[2]) : 0;
    cl_device_id device = selectDevice(platformIndex, platforms, platformCount);

    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    assert(err == CL_SUCCESS);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    assert(err == CL_SUCCESS);

    // Load and build kernel
    std::string kernelSource = loadKernelSource(KERNEL_FILE);
    cl_program program = buildProgram(context, device, kernelSource);

    // Load input image
    cv::Mat sourceRGBA = loadAndConvertImage(IMAGES_MAP.at(args[1]));
    int width = sourceRGBA.cols;
    int height = sourceRGBA.rows;

    cl_image_desc desc = {};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    desc.image_height = height;

    cl_image_format format = {CL_RGBA, CL_UNORM_INT8};
    cl_mem image_in = createImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, desc, sourceRGBA.data);
    cl_mem mask_image = createImage(context, CL_MEM_READ_WRITE, format, desc);
    cl_mem hsv_image = createImage(context, CL_MEM_READ_WRITE, format, desc);

    // hsv_binary_filter kernel
    cl_kernel hsv_kernel = clCreateKernel(program, "hsv_binary_filter", &err);
    clSetKernelArg(hsv_kernel, 0, sizeof(cl_mem), &image_in);
    clSetKernelArg(hsv_kernel, 1, sizeof(cl_mem), &mask_image);
    clSetKernelArg(hsv_kernel, 2, sizeof(cl_mem), &hsv_image);

    size_t globalSize[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    clEnqueueNDRangeKernel(queue, hsv_kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    writeImage(IMAGES + "mask_image.jpg", queue, mask_image, width, height);
    writeImage(IMAGES + "hsv_image.jpg", queue, hsv_image, width, height);

    // Superpixel clustering
    int numClusters = 500;
    const float m = 10.0f;
    std::vector<float> clusterData(numClusters * 5);
    createInitialClusters(width, height, numClusters, clusterData);

    cl_mem clusterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * clusterData.size(), clusterData.data(), &err);
    cl_mem labelBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * width * height, nullptr, &err);
    cl_mem distanceBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * width * height, nullptr, &err);

    cl_kernel cluster_kernel = clCreateKernel(program, "assignPixelsToClusters", &err);
    cl_kernel update_kernel = clCreateKernel(program, "updateClusters", &err);

    std::vector<int> labels(width * height);
    clEnqueueReadBuffer(queue, labelBuffer, CL_TRUE, 0, sizeof(int) * labels.size(), labels.data(), 0, nullptr, nullptr);

    std::vector<float> distances(width * height, FLT_MAX);
    clEnqueueWriteBuffer(queue, distanceBuffer, CL_TRUE, 0, sizeof(float) * distances.size(), distances.data(), 0, nullptr, nullptr);

    std::vector<int> initLabels(width * height, -1);
    clEnqueueWriteBuffer(queue, labelBuffer, CL_TRUE, 0, sizeof(int) * initLabels.size(), initLabels.data(), 0, nullptr, nullptr);

    cv::Mat labelImg(height, width, CV_8UC1);
    for (int i = 0; i < labels.size(); ++i) labelImg.data[i] = static_cast<uchar>((labels[i] * 17) % 255);
    cv::imwrite(IMAGES + "label_image.jpg", labelImg);

    std::vector<float> clusterSums(numClusters * 5, 0.0f);
    std::vector<int> clusterCounts(numClusters, 0);
    cl_mem clusterSumBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                             sizeof(float) * clusterSums.size(), clusterSums.data(), &err);
    cl_mem clusterCountBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                               sizeof(int) * clusterCounts.size(), clusterCounts.data(), &err);

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        TRACE("Clustering iteration %d", iter + 1);

        // Reset accumulation buffers
        std::fill(clusterSums.begin(), clusterSums.end(), 0.0f);
        std::fill(clusterCounts.begin(), clusterCounts.end(), 0);
        clEnqueueWriteBuffer(queue, clusterSumBuffer, CL_TRUE, 0, sizeof(float) * clusterSums.size(), clusterSums.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, clusterCountBuffer, CL_TRUE, 0, sizeof(int) * clusterCounts.size(), clusterCounts.data(), 0, nullptr, nullptr);

        // Assign pixels
        runAssignPixelsToClusters(queue, cluster_kernel, image_in, width, height, clusterBuffer, numClusters, m, labelBuffer, distanceBuffer);

        // Update clusters
        runUpdateClusters(queue, update_kernel, image_in, labelBuffer, width, height, numClusters, clusterSumBuffer, clusterCountBuffer);

        // Read back and update cluster centers
        clEnqueueReadBuffer(queue, clusterSumBuffer, CL_TRUE, 0, sizeof(float) * clusterSums.size(), clusterSums.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, clusterCountBuffer, CL_TRUE, 0, sizeof(int) * clusterCounts.size(), clusterCounts.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, labelBuffer, CL_TRUE, 0, sizeof(int) * labels.size(), labels.data(), 0, nullptr, nullptr);
        std::string outputPath = IMAGES + "superpixel_regions_iter_" + std::to_string(iter + 1) + ".jpg";
        visualizeLabelBoundaries(IMAGES_MAP.at(args[1]), outputPath, labels, width, height, numClusters);

        for (int i = 0; i < numClusters; ++i) {
            int count = clusterCounts[i];
            if (count > 0) {
                clusterData[i * 5 + 0] = clusterSums[i * 5 + 0] / count;
                clusterData[i * 5 + 1] = clusterSums[i * 5 + 1] / count;
                clusterData[i * 5 + 2] = clusterSums[i * 5 + 2] / count;
                clusterData[i * 5 + 3] = clusterSums[i * 5 + 3] / count;
                clusterData[i * 5 + 4] = clusterSums[i * 5 + 4] / count;
            }
        }

        clEnqueueWriteBuffer(queue, clusterBuffer, CL_TRUE, 0, sizeof(float) * clusterData.size(), clusterData.data(), 0, nullptr, nullptr);
        clFinish(queue);
    }

    // Cleanup
    TRACE("Releasing resources");
    clReleaseMemObject(clusterSumBuffer);
    clReleaseMemObject(clusterCountBuffer);
    clReleaseMemObject(clusterBuffer);
    clReleaseMemObject(labelBuffer);
    clReleaseMemObject(distanceBuffer);
    clReleaseMemObject(image_in);
    clReleaseMemObject(mask_image);
    clReleaseMemObject(hsv_image);

    clReleaseKernel(hsv_kernel);
    clReleaseKernel(cluster_kernel);
    clReleaseKernel(update_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    TRACE("PR25LAAW05_SUPERPIXEL application finished");
    return 0;
}