//------------------------------------------------------------------------------
//
// kernel:  vadd  
//
// Purpose: Compute the elementwise sum c = a+b
// 
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//

// __kernel void vector_sum(                             
//    __global float* a,                      
//    __global float* b,                      
//    __global float* c,
//    const int count)               
// {                                          
//    int i = get_global_id(0);               
//    if(i < count)  {
//        c[i] = a[i] + b[i];                 
//    }
// }

// __kernel void make_black_image(read_only image2d_t imgSrc,
//                                write_only image2d_t imgDst)
// {
//    int2 pos = (int2)(get_global_id(0), get_global_id(1));
//    float4 pixel = read_imagef(imgSrc, pos);
   
//    if(pos.x == 0 && pos.y == 0)
//    {
//       printf("Reading %f %f %f %f\n", pixel.x, pixel.y, pixel.z, pixel.w);
//    }

//    pixel.x = 1.0f;
//    pixel.y = 1.0f;
//    pixel.z = 1.0f;
//    pixel.w = 1.0f;
//    if (pos.x == 0 && pos.y == 0)
//    {
//       printf("Writing %f %f %f %f\n", pixel.x, pixel.y, pixel.z, pixel.w);
//    }
//    write_imagef(imgDst, pos, pixel);
// }


// __kernel void increment_pixel(read_only image2d_t imgSrc,
//                                write_only image2d_t imgDst)
// {
//    int2 pos = (int2)(get_global_id(0), get_global_id(1));
//    float4 pixel = read_imagef(imgSrc, pos);
   
//    if(pos.x == 0 && pos.y == 0)
//    {
//       printf("Reading %f %f %f %f\n", pixel.x, pixel.y, pixel.z, pixel.w);
//    }

//    pixel.x = pixel.x - 0.01f;
//    pixel.y = pixel.y - 0.01f;
//    pixel.z = pixel.z - 0.01f;
//    pixel.w = pixel.w - 0.01f;
   
//    if(pos.x == 0 && pos.y == 0)
//    {
//       printf("Writing %f %f %f %f\n", pixel.x, pixel.y, pixel.z, pixel.w);
//    }
//    write_imagef(imgDst, pos, pixel);
// }

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

__kernel void hsv_binary_filter(read_only image2d_t inputImage,
                                write_only image2d_t MaskImage,
                                write_only image2d_t HSVImage) 
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 rgba = read_imagef(inputImage, imageSampler, coord);
    float4 hsv = rgb_to_hsv(rgba.xyz);
    write_imagef(HSVImage, coord, hsv);

    const float h_min = 0.22f;
    const float h_max = 0.33f;
    const float s_min = 0.1f;
    const float v_min = 0.3f;

    int is_green = hsv.x >= h_min && hsv.x <= h_max &&
                   hsv.y >= s_min && hsv.z >= v_min;

    float4 output = (float4)(is_green, is_green, is_green, 1.0f);
    write_imagef(MaskImage, coord, output);
}

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
        float Hc = clusters[c * 5 + 0];
        float Sc = clusters[c * 5 + 1];
        float Vc = clusters[c * 5 + 2];
        float xc = clusters[c * 5 + 3];
        float yc = clusters[c * 5 + 4];

        // Color distance in HSV
        float dh = H - Hc;
        float ds = S - Sc;
        float dv = V - Vc;
        float dc = 1000 * (dh * dh + ds * ds + dv * dv);

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


__kernel void updateClusters(
    read_only image2d_t hsvImage,            // Combined HSV image
    __global const int* labels,              // Assigned labels per pixel
    const int width,
    const int height,
    const int numClusters,
    __global float* clusterSums,             // [numClusters * 5] (H, S, V, x, y)
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
    float H = hsv.x;
    float S = hsv.y;
    float V = hsv.z;

    int base = label * 5;

    clusterSums[base + 0] += H;
    clusterSums[base + 1] += S;
    clusterSums[base + 2] += V;
    clusterSums[base + 3] += (float)x;
    clusterSums[base + 4] += (float)y;

    atomic_inc(&clusterCounts[label]);
}