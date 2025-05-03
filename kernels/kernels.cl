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

__kernel void vector_sum(                             
   __global float* a,                      
   __global float* b,                      
   __global float* c,
   const int count)               
{                                          
   int i = get_global_id(0);               
   if(i < count)  {
       c[i] = a[i] + b[i];                 
   }
}

__kernel void make_black_image(read_only image2d_t imgSrc,
                               write_only image2d_t imgDst)
{
   int2 pos = (int2)(get_global_id(0), get_global_id(1));
   float4 pixel = read_imagef(imgSrc, pos);
   
   if(pos.x == 0 && pos.y == 0)
   {
      printf("Reading %f %f %f %f\n", pixel.x, pixel.y, pixel.z, pixel.w);
   }

   pixel.x = 1.0f;
   pixel.y = 1.0f;
   pixel.z = 1.0f;
   pixel.w = 1.0f;
   if (pos.x == 0 && pos.y == 0)
   {
      printf("Writing %f %f %f %f\n", pixel.x, pixel.y, pixel.z, pixel.w);
   }
   write_imagef(imgDst, pos, pixel);
}


__kernel void increment_pixel(read_write image2d_t img)
{
   int2 pos = (int2)(get_global_id(0), get_global_id(1));
   float4 pixel = read_imagef(img, pos);
   
   if(pos.x == 0 && pos.y == 0)
   {
      printf("Reading %f %f %f %f\n", pixel.x, pixel.y, pixel.z, pixel.w);
   }

   pixel.x = pixel.x - 0.01f;
   pixel.y = pixel.y - 0.01f;
   pixel.z = pixel.z - 0.01f;
   pixel.w = pixel.w - 0.01f;
   
   if(pos.x == 0 && pos.y == 0)
   {
      printf("Writing %f %f %f %f\n", pixel.x, pixel.y, pixel.z, pixel.w);
   }
   write_imagef(img, pos, pixel);
}

