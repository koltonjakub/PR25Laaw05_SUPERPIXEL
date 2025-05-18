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


// __kernel void increment_pixel(read_write image2d_t img)
// {
//    int2 pos = (int2)(get_global_id(0), get_global_id(1));
//    float4 pixel = read_imagef(img, pos);
   
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
//    write_imagef(img, pos, pixel);
// }

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

float3 rgb_to_hsv(float3 c) {
   float4 K = (float4)(0.0f, -1.0f/3.0f, 2.0f/3.0f, -1.0f);
   float4 p = c.g < c.b ? (float4)(c.bg, K.wz) : (float4)(c.gb, K.xy);
   float4 q = c.r < p.x ? (float4)(p.xyw, c.r) : (float4)(c.r, p.yzx);

   float d = q.x - min(q.w, q.y);
   float e = 1.0e-10f;

   float h = fabs(q.z + (q.w - q.y) / (6.0f * d + e));
   float s = d / (q.x + e);
   float v = q.x;

   return (float3)(h, s, v);
}

__kernel void hsv_binary_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage) {
   const int2 coord = (int2)(get_global_id(0), get_global_id(1));
   float4 rgba = read_imagef(inputImage, imageSampler, coord);

   float3 hsv = rgb_to_hsv(rgba.xyz);

   // Zakres koloru zielonego i żółto-zielonego w HSV:
   // Hue w [0.22, 0.33], Saturation i Value powyżej minimalnego progu
   const float h_min = 0.05f; // około 80° (zielony)
   const float h_max = 0.60f; // około 120° (żółto-zielony)
   const float s_min = 0.1f;
   const float v_min = 0.3f;

   int is_green = hsv.x >= h_min && hsv.x <= h_max &&
                  hsv.y >= s_min &&
                  hsv.z >= v_min;

   // Binarna maska: biały jeśli w zakresie, czarny jeśli nie
   float4 output = (float4)(is_green, is_green, is_green, 1);
   // float4 output = (float4)(hsv.x, hsv.y, hsv.z, 1.0f);

   write_imagef(outputImage, coord, output);
}