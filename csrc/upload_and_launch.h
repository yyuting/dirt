#include <chrono>

#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glext.h>

#undef Status
#undef Success
#undef None

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/stream_executor.h"

// This form of include path matches what the TensorFlow headers use
#include <cuda/include/cuda.h>
#include <cuda/include/cuda_runtime_api.h>
#include <cuda/include/cuda_gl_interop.h>

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"

void launch_background_upload(
    cudaArray_t &dest_array, tensorflow::Tensor const &src_tensor,
    int const dest_height, int const dest_width,
    Eigen::GpuDevice const &device
);

void launch_pixels_download(
    tensorflow::Tensor &dest_tensor, cudaArray_t const &src_array,
    int const src_height, int const src_width,
    Eigen::GpuDevice const &device
);