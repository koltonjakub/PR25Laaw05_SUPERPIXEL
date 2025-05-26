/**
 * @file        main.cpp
 * @brief       PR25LAAW05_SUPERPIXEL application.
 * @author      Jan Rosa, Karolina Piotrowska, Jakub Kołton
 * @date        2025-05-4
 * @version     1.0
 *
 * @details
 * This file contains the application responsible for orchestrating the 
 * superpixel segmentation pipeline using OpenCL. The program supports batch 
 * and single-image processing with optional tuning of clustering parameters.
 *
 * Key features:
 * - OpenCL accelerated image conversion and clustering.
 * - HSV binary filtering.
 * - Iterative superpixel refinement.
 * - Visual output with superpixel boundary overlays.
 * - Command-line interface for flexible usage.
 */

#include <CL/cl.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <queue>
#include <chrono>
#include <sstream>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define TRACE(fmt, ...)                                                            \
    // { printf("[TRACE_FMT] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); };

const std::string REPOROOT = std::filesystem::current_path().parent_path().string() + "/PR25Laaw05_SUPERPIXEL";
// const std::string REPOROOT = std::filesystem::current_path().parent_path().string();
const std::string KERNELS = REPOROOT + "/kernels/";
const std::string KERNEL_FILE = KERNELS + "kernels.cl";
const std::string IMAGES = REPOROOT + "/images/";
const std::string INPUT_DIR = REPOROOT + "/images/input/";
const std::string OUTPUT_DIR = REPOROOT + "/images/ouput/";
const int NUM_ITERATIONS = 10;
const int NUM_SUPERPIXELS = 100;

/**
 * @brief Loads the contents of an OpenCL kernel source file into a string.
 *
 * This function opens the file specified by the given file path, reads its entire
 * contents, and returns it as a standard string.
 *
 * @param filePath The path to the kernel source file.
 * @return A string containing the full contents of the kernel source file.
 *
 * @throws std::runtime_error If the file cannot be opened.
 */
std::string loadKernelSource(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) throw std::runtime_error("Failed to open kernel file: " + filePath);
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

/**
 * @brief Selects the first available OpenCL device from a specified platform.
 *
 * This function retrieves the list of OpenCL devices associated with the platform
 * at the given index and returns the first available device. It supports all device
 * types (CPU, GPU, etc.).
 *
 * @param platformIndex The index of the desired platform in the `platforms` array.
 * @param platforms Pointer to an array of OpenCL platform IDs.
 * @param platformCount The total number of available platforms.
 * @return The first available OpenCL device (`cl_device_id`) on the specified platform.
 *
 * @throws std::runtime_error If the platform index is out of bounds or if no devices are found on the selected platform.
 */
cl_device_id selectDevice(int platformIndex, cl_platform_id* platforms, unsigned int platformCount) {
    if (platformIndex >= static_cast<int>(platformCount)) throw std::runtime_error("Invalid platform index.");
    cl_device_id devices[64];
    unsigned int deviceCount;
    cl_int result = clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_ALL, 64, devices, &deviceCount);
    if (result != CL_SUCCESS || deviceCount == 0) throw std::runtime_error("No OpenCL devices found.");
    return devices[0]; // Default: first device
}

/**
 * @brief Loads an image from a file and converts it to RGBA format.
 *
 * This function reads an image from the specified file path using OpenCV,
 * verifies that the image was successfully loaded, and then converts it from
 * BGR (the default format returned by OpenCV) to RGBA format.
 *
 * @param path The path to the image file.
 * @return A cv::Mat object containing the image in RGBA format.
 *
 * @throws std::runtime_error If the image cannot be loaded (e.g., file not found or unsupported format).
 */
cv::Mat loadAndConvertImage(const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) throw std::runtime_error("Failed to load image.");
    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
    return rgba;
}

/**
 * @brief Creates an OpenCL image object.
 *
 * This function wraps the `clCreateImage` API call to create an OpenCL image
 * with the specified context, memory flags, format, description, and optional host pointer.
 *
 * @param context The OpenCL context in which to create the image.
 * @param flags Memory flags specifying allocation and usage (e.g., `CL_MEM_READ_ONLY`, `CL_MEM_COPY_HOST_PTR`).
 * @param format The image format descriptor (e.g., channel order and data type).
 * @param desc The image description, including type, dimensions, and layout.
 * @param hostPtr Optional pointer to the host memory to use as the backing store.
 * @return A `cl_mem` handle representing the created OpenCL image.
 *
 * @note The function uses an assertion to ensure that image creation succeeded. If `CL_SUCCESS` is not returned, the program will abort in debug builds.
 */
cl_mem createImage(cl_context context, cl_mem_flags flags, cl_image_format format,
                   cl_image_desc desc, void* hostPtr = nullptr) {
    cl_int result;
    cl_mem image = clCreateImage(context, flags, &format, &desc, hostPtr, &result);
    assert(result == CL_SUCCESS);
    return image;
}

/**
 * @brief Creates and builds an OpenCL program from source code.
 *
 * This function creates an OpenCL program object from the given source code string
 * and compiles it for the specified device. If the build fails, the build log is
 * printed to standard error and the application exits.
 *
 * @param context The OpenCL context in which to create the program.
 * @param device The OpenCL device for which the program will be built.
 * @param source The source code of the OpenCL program as a string.
 * @return A `cl_program` object representing the built OpenCL program.
 *
 * @note If program creation fails, an `assert` is triggered. If program build fails,
 * the build log is printed and the application terminates with `exit(-1)`.
 */
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

/**
 * @brief Reads an OpenCL image object and writes it to a file using OpenCV.
 *
 * This function reads pixel data from a 2D OpenCL image object into a `cv::Mat`,
 * then saves it to disk at the specified file path using OpenCV's `imwrite`.
 *
 * @param path The file path where the image will be saved.
 * @param queue The OpenCL command queue used to enqueue the read operation.
 * @param image The OpenCL image object (`cl_mem`) to be read.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @param type (Optional) OpenCV image type (e.g., `CV_8UC4`). Defaults to `CV_8UC4` (8-bit RGBA).
 *
 * @note The function uses `assert` to ensure that the image read operation was successful.
 * If the assertion fails, the application will terminate in debug builds.
 */
void writeImage(const std::string& path, cl_command_queue queue, cl_mem image,
                int width, int height, int type = CV_8UC4) {
    cv::Mat output(height, width, type);
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {static_cast<size_t>(width), static_cast<size_t>(height), 1};
    cl_int res = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0, output.data, 0, nullptr, nullptr);
    assert(res == CL_SUCCESS);
    cv::imwrite(path, output);
}

/**
 * @brief Extracts an OpenCL 2D image into a cv::Mat.
 *
 * This function reads pixel data from a 2D OpenCL image (`cl_mem`) and
 * returns it as an OpenCV `cv::Mat`.
 *
 * @param queue The OpenCL command queue used to read the image.
 * @param image The OpenCL image object to be read.
 * @param width The image width in pixels.
 * @param height The image height in pixels.
 * @param type (Optional) OpenCV image type (default: CV_8UC4 for RGBA).
 * @return cv::Mat The image data as an OpenCV matrix.
 */
cv::Mat extractImage(cl_command_queue queue, cl_mem image,
                     int width, int height, int type = CV_8UC4) {
    cv::Mat output(height, width, type);
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {static_cast<size_t>(width), static_cast<size_t>(height), 1};

    cl_int err = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0, output.data, 0, nullptr, nullptr);
    assert(err == CL_SUCCESS && "Failed to read OpenCL image");

    return output;
}

/**
 * @brief Initializes cluster centers on a grid over the image, guided by a binary mask.
 *
 * This function places cluster centers (typically for image segmentation or clustering tasks)
 * across the input image in a uniform grid layout. The number of clusters may be adjusted
 * to fit a complete grid. The placement is guided by a binary mask image: clusters are
 * placed preferentially in regions where the mask is non-zero, with sparse fallback
 * sampling in unmasked regions.
 *
 * Each cluster center stores five values: H, S, V color components (defaulted to 0.5),
 * and the (x, y) coordinates in the image.
 *
 * @param width Width of the target image in pixels.
 * @param height Height of the target image in pixels.
 * @param numClusters Input/output parameter. Initially specifies the desired number of clusters;
 *                    will be adjusted to match the number of grid cells.
 * @param clusterData Output vector to store cluster descriptors (H, S, V, x, y) per cluster.
 * @param MaskImagePath Path to a grayscale mask image. Non-zero pixels indicate preferred
 *                      regions for cluster placement.
 *
 * @note If the mask image cannot be loaded, the function prints an error and returns early.
 * @note Sparse clusters are still added in blank regions using a sampling pattern (every 8th grid cell).
 */
void createInitialClusters(int width, int height, int& numClusters, std::vector<float>& clusterData, cv::Mat mask_image_mat) {
    // Estimate grid dimensions based on aspect ratio and target cluster count
    int gridCols = static_cast<int>(std::sqrt((float)numClusters * width / height));
    int gridRows = static_cast<int>(std::ceil((float)numClusters / gridCols));

    // Override number of clusters to fit full grid
    numClusters = gridCols * gridRows;
    clusterData.clear();
    clusterData.reserve(numClusters * 5);  // H, S, V, x, y

    float stepX = static_cast<float>(width) / gridCols;
    float stepY = static_cast<float>(height) / gridRows;

    const int blank_divider = 8;

    cv::Mat mask_gray;
    cv::cvtColor(mask_image_mat, mask_gray, cv::COLOR_RGBA2GRAY);

    int c = 0;
    for (int row = 0; row < gridRows; ++row) {
        for (int col = 0; col < gridCols; ++col) {
            float cx = (col + 0.5f) * stepX;
            float cy = (row + 0.5f) * stepY;

            // Clamp to image bounds
            if (cx >= width) cx = width - 1;
            if (cy >= height) cy = height - 1;

            //fix this code
            if (mask_gray.at<uchar>(cy, cx) > 0)
            {
                clusterData.push_back(0.5f);      // H
                clusterData.push_back(0.5f);      // S
                clusterData.push_back(0.5f);      // V
                clusterData.push_back(cx);        // X
                clusterData.push_back(cy);        // Y
            }
            else if (!(static_cast<int>(col)%blank_divider || static_cast<int>(row)%blank_divider))
            {
                //Less clusters for no leafes
                clusterData.push_back(0.5f);      // H
                clusterData.push_back(0.5f);      // S
                clusterData.push_back(0.5f);      // V
                clusterData.push_back(cx);        // X
                clusterData.push_back(cy);        // Y
            }//else do nothing, no cluster
            
            ++c;
        }
    }

    TRACE("Adjusted numClusters = %d (%d × %d grid)", numClusters, gridCols, gridRows);
}

/**
 * @brief Checks the result of an OpenCL operation and exits on failure.
 *
 * Prints an error message and terminates the program if the given result
 * is not `CL_SUCCESS`.
 *
 * @param result The OpenCL error code to check.
 * @param message A message to display if the result indicates failure.
 */
void assertCLSuccess(cl_int result, const char* message) {
    if (result != CL_SUCCESS) {
        std::cerr << message << " Error Code: " << result << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Runs an OpenCL kernel to assign each pixel to the nearest cluster.
 *
 * Sets kernel arguments and launches a 2D kernel over the image to compute
 * pixel-to-cluster assignments based on color and spatial proximity.
 *
 * @param queue OpenCL command queue.
 * @param kernel OpenCL kernel for assignment.
 * @param hsv_image HSV image buffer.
 * @param width Image width.
 * @param height Image height.
 * @param clusterBuffer Buffer with cluster data.
 * @param numClusters Number of clusters.
 * @param m Compactness factor.
 * @param labelBuffer Output buffer for pixel labels.
 * @param distanceBuffer Output buffer for distances.
 */
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
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    assertCLSuccess(err, "Kernel enqueue failed");
    clFinish(queue);
}

/**
 * @brief Runs an OpenCL kernel to update cluster sums and counts based on pixel assignments.
 *
 * Sets kernel arguments and launches a 2D kernel that accumulates HSV values and counts
 * for each cluster from the labeled pixels.
 *
 * @param queue OpenCL command queue.
 * @param updateKernel Kernel to update cluster data.
 * @param hsv_image HSV image buffer.
 * @param labelBuffer Buffer with pixel cluster labels.
 * @param width Image width.
 * @param height Image height.
 * @param numClusters Number of clusters.
 * @param clusterSumBuffer Buffer to accumulate cluster HSV sums.
 * @param clusterCountBuffer Buffer to accumulate cluster pixel counts.
 */
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
    cl_int err = clEnqueueNDRangeKernel(queue, updateKernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    assertCLSuccess(err, "Kernel enqueue failed");
    clFinish(queue);
}

/**
 * @brief Visualizes cluster boundaries by overlaying them on the original image.
 *
 * Loads an input image and highlights the boundaries between different cluster labels
 * by coloring boundary pixels in red. The resulting image is saved to the specified output path.
 *
 * @param inputImagePath Path to the input image file.
 * @param outputImagePath Path where the output image with boundaries will be saved.
 * @param labels A vector of cluster labels for each pixel (size = width * height).
 * @param width Width of the image in pixels.
 * @param height Height of the image in pixels.
 * @param numClusters Number of clusters (not used directly but kept for interface consistency).
 *
 * @note The function assumes `labels` are arranged in row-major order.
 * @note If the input image cannot be loaded or dimensions mismatch, the function prints an error and returns.
 */
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

/**
 * @brief Recursively finds image files in a directory with common image extensions.
 *
 * Searches the given directory and all its subdirectories for files with extensions
 * matching common image formats (.jpg, .jpeg, .png, .bmp, .tif, .tiff).
 *
 * @param directoryPath The path to the directory to search.
 * @return A vector of strings containing full file paths to the found image files.
 */
std::vector<std::string> findImageFiles(const std::string& directoryPath) {
    std::vector<std::string> imageFiles;
    std::vector<std::string> imageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff" };

    for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (std::find(imageExtensions.begin(), imageExtensions.end(), extension) != imageExtensions.end()) {
                imageFiles.push_back(entry.path().string());
            }
        }
    }

    return imageFiles;
}

/**
 * @brief Constructs an output file path in an 'output' directory relative to the input file.
 *
 * Given an input file path, this function generates a corresponding output file path
 * by placing the output in a sibling directory named "output" and appending "_out.png"
 * to the input file's base name.
 *
 * Example: For input "/path/to/image.jpg", the output path will be "/path/output/image_out.png".
 *
 * @param path The input file path.
 * @return The constructed output file path as a string.
 */
std::string getOutputFilePath(const std::string& path) {
    std::filesystem::path inputPath(path);
    std::filesystem::path outputDir = inputPath.parent_path() / "../output";

    std::string baseName = inputPath.stem().string();  // e.g., "image"
    std::string newFilename = baseName + "_out.png";   // e.g., "image_out.png"

    std::filesystem::path fullOutputPath = outputDir / newFilename;
    return fullOutputPath.lexically_normal().string();
}

/**
 * @brief Entry point of the PR25LAAW05_SUPERPIXEL application.
 *
 * This application performs superpixel segmentation on a single image or a batch of images
 * using OpenCL acceleration. It converts input images to HSV format, applies a binary filter,
 * and then clusters pixels into superpixels using an iterative approach. The boundaries of
 * superpixels are then visualized and saved.
 *
 * @param argc The number of command-line arguments.
 * @param args The array of command-line arguments:
 * - `args[1]` - Number of images to process (0 for all images in `INPUT_DIR`, >0 for specific images).
 * - `args[2]` - OpenCL platform index to use.
 * - `args[3]` - (Optional) Number of clustering cycles (default: `NUM_ITERATIONS`).
 * - `args[4]` - (Optional) Expected number of superpixels/clusters (default: `NUM_SUPERPIXELS`).
 * - `args[5]` - (Optional) Compactness factor for clustering (default: 10.0f).
 * - `args[6+]` - (Optional) Image filenames if `imgs_quantity > 0`.
 *
 * @return Returns 0 on successful completion.
 *
 * @details
 * Steps performed:
 * - Parses command-line arguments.
 * - Initializes OpenCL platform, context, and device.
 * - Loads and builds the OpenCL kernel program.
 * - For each image:
 *   - Loads image and converts it to HSV space using an OpenCL kernel.
 *   - Applies a binary HSV filter to generate a mask.
 *   - Initializes clusters and iteratively refines them based on pixel distance and color similarity.
 *   - Visualizes superpixel boundaries and saves the output.
 *   - Measures and logs processing time for each step.
 * - Releases OpenCL resources after processing.
 */
int main(int argc, char* args[]) {
    TRACE("PR25LAAW05_SUPERPIXEL application started");
    auto program_start = std::chrono::high_resolution_clock::now();

    std::ostringstream oss;
    oss << "Logging Timing Raport\n";

    int imgs_quantity = std::stoi(args[1]);  // 0 for all in input dir
    int platformIndex = std::stoi(args[2]);  //Platfrom dependent
    const int clusteringCycles = (argc > 3) ? std::stoi(args[3]) : NUM_ITERATIONS; 
    int expected_numClusters = (argc > 4) ? std::stoi(args[4]) : NUM_SUPERPIXELS;
    const float compactness_factor = static_cast<float>((argc > 5) ? std::stoi(args[5]) : 10.0f);

    TRACE("Operation type: %s", imgs_quantity ? "all images" : "one image");
    std::queue<std::string> images_path_queue;
    if (imgs_quantity != 0) // Just one image
    {
        for (size_t i = 0; i < MIN(imgs_quantity, argc-6); ++i)
        {
            images_path_queue.push(INPUT_DIR + args[6+i]);
        }
    }
    else
    {
        std::vector<std::string> files = findImageFiles(INPUT_DIR);
        for (const std::string& file : files) {
            images_path_queue.push(file);
        }
    }

    oss << "Images quantity: " << imgs_quantity << "\n";
    oss << "Platform index: " << platformIndex << "\n";
    oss << "Clustering cycles: " << clusteringCycles << "\n";
    oss << "Expected number of clusters: " << expected_numClusters << "\n";
    oss << "Compactness factor: " << compactness_factor << "\n";
    
    // OpenCL Setup
    TRACE("Initializing Platform");
    cl_platform_id platforms[64];
    unsigned int platformCount;
    clGetPlatformIDs(64, platforms, &platformCount);

    for (unsigned int i = 0; i < platformCount; ++i) {
        char name[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        TRACE("Platform %u: %s", i, name);
    }

    cl_device_id device = selectDevice(platformIndex, platforms, platformCount);

    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    assert(err == CL_SUCCESS);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    assert(err == CL_SUCCESS);

    // Load and build kernel
    TRACE("Loading Kernel");
    std::string kernelSource = loadKernelSource(KERNEL_FILE);
    cl_program program = buildProgram(context, device, kernelSource);

    // Load input image
    TRACE("Entering Dequeuing loop");
    while (!images_path_queue.empty())
    {
        auto currentImagePath = images_path_queue.front();
        images_path_queue.pop();
        std::string outputFilePath = getOutputFilePath(currentImagePath);

        TRACE("Dequeuing %s", currentImagePath.c_str());
        oss << "\tProcessing " << currentImagePath << '\n';

        auto image_start = std::chrono::high_resolution_clock::now();
        cv::Mat sourceRGBA = loadAndConvertImage(currentImagePath);
        int width = sourceRGBA.cols;
        int height = sourceRGBA.rows;

        cl_image_desc desc = {};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = width;
        desc.image_height = height;

        cl_image_format format = {CL_RGBA, CL_UNORM_INT8};
        TRACE("Creating input images images (sic!)");
        cl_mem image_in = createImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, desc, sourceRGBA.data);
        cl_mem mask_image = createImage(context, CL_MEM_READ_WRITE, format, desc);
        cl_mem hsv_image = createImage(context, CL_MEM_READ_WRITE, format, desc);

        // hsv_binary_filter kernel
        TRACE("Initializing HSV conversion kernel");
        cl_kernel hsv_kernel = clCreateKernel(program, "hsv_binary_filter", &err);
        clSetKernelArg(hsv_kernel, 0, sizeof(cl_mem), &image_in);
        clSetKernelArg(hsv_kernel, 1, sizeof(cl_mem), &mask_image);
        clSetKernelArg(hsv_kernel, 2, sizeof(cl_mem), &hsv_image);

        size_t globalSize[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
        err = clEnqueueNDRangeKernel(queue, hsv_kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
        assertCLSuccess(err, "Kernel enqueue failed");
        clFinish(queue);

        // TRACE("Writing HSV and mask");
        // writeImage(IMAGES + "mask_image.jpg", queue, mask_image, width, height);
        // writeImage(IMAGES + "hsv_image.jpg", queue, hsv_image, width, height);
        TRACE("Getting HSV and mask");
        cv::Mat mask_image_mat = extractImage(queue, mask_image, width, height);
        cv::Mat hsv_image_mat = extractImage(queue, hsv_image, width, height);

        TRACE("Initializing Superpixels Structures");
        int numClusters = expected_numClusters;
        std::vector<float> clusterData(0);
        createInitialClusters(width, height, numClusters, clusterData, mask_image_mat);

        cl_mem clusterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(float) * clusterData.size(), clusterData.data(), &err);
        cl_mem labelBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * width * height, nullptr, &err);
        cl_mem distanceBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * width * height, nullptr, &err);

        TRACE("Creating Superpixel kernels");
        cl_kernel cluster_kernel = clCreateKernel(program, "assignPixelsToClusters", &err);
        cl_kernel update_kernel = clCreateKernel(program, "updateClusters", &err);

        TRACE("Width: %d, Height: %d", width, height);

        std::vector<int> labels(width * height, 0);
        clEnqueueReadBuffer(queue, labelBuffer, CL_TRUE, 0, sizeof(int) * labels.size(), labels.data(), 0, nullptr, nullptr);

        std::vector<float> distances(width * height, FLT_MAX);
        clEnqueueWriteBuffer(queue, distanceBuffer, CL_TRUE, 0, sizeof(float) * distances.size(), distances.data(), 0, nullptr, nullptr);

        std::vector<int> initLabels(width * height, -1);
        clEnqueueWriteBuffer(queue, labelBuffer, CL_TRUE, 0, sizeof(int) * initLabels.size(), initLabels.data(), 0, nullptr, nullptr);

        std::vector<int> clusterSums(numClusters * 5, 0); // Use integers for clusterSums
        cl_mem clusterSumBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                sizeof(int) * clusterSums.size(), clusterSums.data(), &err);
        assertCLSuccess(err, "Failed to create clusterSums buffer");

        std::vector<int> clusterCounts(numClusters, 0);
        cl_mem clusterCountBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                sizeof(int) * clusterCounts.size(), clusterCounts.data(), &err);

        for (int iter = 0; iter < clusteringCycles; ++iter) {
            TRACE("Clustering iteration %d", iter + 1);
            oss << "\t\t Clusterig iteration " << iter + 1 << "\n";

            auto cycle_start = std::chrono::high_resolution_clock::now();

            // Reset accumulation buffers
            std::fill(clusterSums.begin(), clusterSums.end(), 0.0f);
            std::fill(clusterCounts.begin(), clusterCounts.end(), 0);
            clEnqueueWriteBuffer(queue, clusterSumBuffer, CL_TRUE, 0, sizeof(int) * clusterSums.size(), clusterSums.data(), 0, nullptr, nullptr);
            clEnqueueWriteBuffer(queue, clusterCountBuffer, CL_TRUE, 0, sizeof(int) * clusterCounts.size(), clusterCounts.data(), 0, nullptr, nullptr);

            // Assign pixels
            runAssignPixelsToClusters(queue, cluster_kernel, image_in, width, height, clusterBuffer, numClusters, compactness_factor, labelBuffer, distanceBuffer);

            // Update clusters
            runUpdateClusters(queue, update_kernel, image_in, labelBuffer, width, height, numClusters, clusterSumBuffer, clusterCountBuffer);

            // Read back and update cluster centers
            clEnqueueReadBuffer(queue, clusterSumBuffer, CL_TRUE, 0, sizeof(int) * clusterSums.size(), clusterSums.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(queue, clusterCountBuffer, CL_TRUE, 0, sizeof(int) * clusterCounts.size(), clusterCounts.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(queue, labelBuffer, CL_TRUE, 0, sizeof(int) * labels.size(), labels.data(), 0, nullptr, nullptr);
            //std::string outputPath = IMAGES + "superpixel_regions_iter_" + std::to_string(iter + 1) + ".jpg";

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
            
            auto cycle_end = std::chrono::high_resolution_clock::now();
            auto cycle_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cycle_end - cycle_start).count();
            TRACE("Clustering iteration %d took %ld ms", iter + 1, cycle_duration);
            oss << "\t\t Duration " << cycle_duration << " ms\n";
            // std::ostringstream oss2;
            // oss2 << outputFilePath.substr(0, outputFilePath.size() - 4) << "_cycle_" << iter << ".png";
            // visualizeLabelBoundaries(currentImagePath, oss2.str(), labels, width, height, numClusters);
        }
        TRACE("Visualizing boundaries")
        visualizeLabelBoundaries(currentImagePath, outputFilePath, labels, width, height, numClusters);
        // Cleanup
        TRACE("Releasing Mem Objects");
        clReleaseMemObject(clusterSumBuffer);
        clReleaseMemObject(clusterCountBuffer);
        clReleaseMemObject(clusterBuffer);
        clReleaseMemObject(labelBuffer);
        clReleaseMemObject(distanceBuffer);
        clReleaseMemObject(image_in);
        clReleaseMemObject(mask_image);
        clReleaseMemObject(hsv_image);
        TRACE("Releasing Kernels");
        clReleaseKernel(hsv_kernel);
        clReleaseKernel(cluster_kernel);
        clReleaseKernel(update_kernel);

        auto image_end = std::chrono::high_resolution_clock::now();
        auto image_duration = std::chrono::duration_cast<std::chrono::milliseconds>(image_end - image_start).count();
        TRACE("Finished processing image %s in %ld ms", currentImagePath.c_str(), image_duration);
        oss << "\tDuration: " << image_duration << " ms\n";
    }

    TRACE("Releasing Other");
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - program_start).count();
    TRACE("Total program execution time: %ld ms", total_duration);
    oss<<"Duration:" << total_duration << "ms\n";
    std::cout << oss.str();

    TRACE("PR25LAAW05_SUPERPIXEL application finished");
    return 0;
}