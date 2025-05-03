# PR25Laaw05_SUPERPIXEL

### Description
Repository for project on Acceleration of Vision Algorithms in GPU Subject: Superpixels in OpenCL.

### Dependencies:

CMake: minimum required version is 3.10, so far tested on 3.28.3
OpenCL:
    ICD loader properties
    ICD loader Name                                 OpenCL ICD Loader
    ICD loader Vendor                               OCL Icd free software
    ICD loader Version                              2.3.2
    ICD loader Profile                              OpenCL 3.0
OpenCV: 4.6.0

### List of compatible drivers:

NVDIA:
    NVIDIA-SMI 550.144.03             Driver Version: 550.144.03     CUDA Version: 12.4

Runnig the application:
    cd <repoot_dir>
    mkdir build (if not yet present)
    cd build
    cmake ..
    make
    (sudo) ./PR25Laaw05_SUPERPIXEL - I had troubles to run this, as well as clinfo and other opencl related commands without sudo on Ubuntu 24.04 LTS / 22.04 LTS, issue lies in the priviliges to the nvidia device within root dir.
