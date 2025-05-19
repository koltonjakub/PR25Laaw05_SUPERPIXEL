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

float3 rgb_to_hsv(float3 c) {
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

    return (float3)(h, s, v);
}

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void hsv_binary_filter(read_only image2d_t inputImage,
                                write_only image2d_t outputImage) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 rgba = read_imagef(inputImage, imageSampler, coord);
    float3 hsv = rgb_to_hsv(rgba.xyz);

    const float h_min = 0.22f;
    const float h_max = 0.33f;
    const float s_min = 0.1f;
    const float v_min = 0.3f;

    int is_green = hsv.x >= h_min && hsv.x <= h_max &&
                   hsv.y >= s_min && hsv.z >= v_min;

    float4 output = (float4)(is_green, is_green, is_green, 1.0f);
    write_imagef(outputImage, coord, output);
}