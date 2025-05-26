/**
 * @brief Converts an RGB color to HSV.
 *
 * Takes a float3 RGB color and returns a float4 HSV representation.
 * Hue is normalized to [0.0, 1.0]. Saturation and value are also in [0.0, 1.0].
 *
 * @param c Input RGB color as float3, with components in [0.0, 1.0].
 * @return float4 HSV color: (h, s, v, 0.0).
 */
float4 rgb_to_hsv(float3 c) {
    float r = c.x;
    float g = c.y;
    float b = c.z;

    float maxc = fmax(fmax(r, g), b);
    float minc = fmin(fmin(r, g), b);
    float delta = maxc - minc;

    float h = 0.0f;
    if (delta > 0.00001f) {
        if (maxc == r) {
            h = (g - b) / delta;
        } else if (maxc == g) {
            h = 2.0f + (b - r) / delta;
        } else {
            h = 4.0f + (r - g) / delta;
        }
        h = h / 6.0f;
        if (h < 0.0f) h += 1.0f;
    }

    float s = (maxc == 0.0f) ? 0.0f : (delta / maxc);
    float v = maxc;

    return (float4)(h, s, v, 0.0);
}

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

/**
 * @brief Converts an input RGBA image to HSV and creates a binary mask.
 *
 * Each pixel is converted from RGB to HSV and written to HSVImage.
 * A binary mask is generated in MaskImage based on HSV thresholds,
 * detecting pixels that meet specific hue, saturation, and value ranges.
 *
 * @param inputImage   Input image in RGBA format.
 * @param MaskImage    Output binary mask image (1.0 for match, 0.0 otherwise).
 * @param HSVImage     Output HSV image.
 */
__kernel void hsv_binary_filter(read_only image2d_t inputImage,
                                write_only image2d_t MaskImage,
                                write_only image2d_t HSVImage) 
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 rgba = read_imagef(inputImage, imageSampler, coord);
    float4 hsv = rgb_to_hsv(rgba.xyz);
    write_imagef(HSVImage, coord, hsv);

    const float h_min = 0.0f;
    const float h_max = 1.0f;
    const float s_min = 0.11f;
    const float v_min = 0.11f;

    int is_green = hsv.x >= h_min && hsv.x <= h_max &&
                   hsv.y >= s_min && hsv.z >= v_min;

    float4 output = (float4)(is_green, is_green, is_green, 1.0f);
    write_imagef(MaskImage, coord, output);
}

const float PRECISION = 100000.0f; // Precision for HSV values

/**
 * @brief Assigns each pixel to the nearest cluster based on HSV and spatial distance.
 *
 * Computes a combined distance metric (color + position) for each pixel to all clusters,
 * assigning it to the one with the lowest value.
 *
 * @param hsvImage   Input HSV image (H, S, V in float4).
 * @param width      Image width.
 * @param height     Image height.
 * @param clusters   Cluster data [H, S, V, x, y] scaled (length: numClusters * 5).
 * @param numClusters Number of clusters.
 * @param m          Compactness factor (controls spatial vs. color influence).
 * @param labels     Output: assigned cluster index per pixel.
 * @param distances  Output: distance to assigned cluster per pixel.
 */
__kernel void assignPixelsToClusters(
    read_only image2d_t hsvImage,       // combined HSV input image
    const int width,
    const int height,
    __global const float* clusters,     // [numClusters * 5] (H,S,V,x,y)
    const int numClusters,
    const float m,                      // compactness factor
    __global int* labels,
    __global float* distances
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Read HSV values from input image (assuming stored as float4 with H,S,V in RGB channels)
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    float4 hsv = read_imagef(hsvImage, sampler, (int2)(x, y));
    float H = hsv.x;
    float S = hsv.y;
    float V = hsv.z;

    float minDist = FLT_MAX;
    int bestCluster = -1;

    for (int c = 0; c < numClusters; ++c) {
        float Hc = clusters[c * 5 + 0] / PRECISION; // Scale back H
        float Sc = clusters[c * 5 + 1] / PRECISION; // Scale back S
        float Vc = clusters[c * 5 + 2] / PRECISION; // Scale back V
        float xc = clusters[c * 5 + 3];           // x remains as is
        float yc = clusters[c * 5 + 4];           // y remains as is

        // Color distance in HSV
        float dh = H - Hc;
        float ds = S - Sc;
        float dv = V - Vc;
        float dc = 10000 * (dh * dh + ds * ds + dv * dv);

        // Spatial distance
        float dx = (float)x - xc;
        float dy = (float)y - yc;
        float ds_xy = dx * dx + dy * dy;

        // Combined distance
        float D = dc + (m / 10.0f) * ds_xy;

        if (D < minDist) {
            minDist = D;
            bestCluster = c;
        }
    }

    labels[idx] = bestCluster;
    distances[idx] = minDist;
}

/**
 * @brief Accumulates HSV and spatial values for each cluster.
 *
 * Each pixel contributes its HSV values and (x, y) position to its assigned cluster.
 * The data is accumulated into `clusterSums` and `clusterCounts` using atomic ops.
 *
 * @param hsvImage       Input HSV image (float4 per pixel).
 * @param labels         Per-pixel cluster labels (size: width * height).
 * @param width          Image width.
 * @param height         Image height.
 * @param numClusters    Number of superpixel clusters.
 * @param clusterSums    Accumulated [H, S, V, x, y] per cluster (scaled ints).
 * @param clusterCounts  Pixel count per cluster.
 *
 * @note HSV values are scaled by PRECISION and summed atomically.
 */
__kernel void updateClusters(
    read_only image2d_t hsvImage,            // Combined HSV image
    __global const int* labels,              // Assigned labels per pixel
    const int width,
    const int height,
    const int numClusters,
    __global int* clusterSums,               // [numClusters * 5] (H, S, V, x, y) as integers
    __global int* clusterCounts              // [numClusters]
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int label = labels[idx];
    if (label < 0 || label >= numClusters) return;

    int2 coord = (int2)(x, y);
    float4 hsv = read_imagef(hsvImage, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, coord);
    int H = (int)(hsv.x * PRECISION); // Scale H by 1000
    int S = (int)(hsv.y * PRECISION); // Scale S by 1000
    int V = (int)(hsv.z * PRECISION); // Scale V by 1000

    int base = label * 5;

    atomic_add(&clusterSums[base + 0], H);
    atomic_add(&clusterSums[base + 1], S);
    atomic_add(&clusterSums[base + 2], V);
    atomic_add(&clusterSums[base + 3], x);
    atomic_add(&clusterSums[base + 4], y);

    atomic_inc(&clusterCounts[label]);
}