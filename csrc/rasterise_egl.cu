
#define GOOGLE_CUDA 1
#define EIGEN_USE_GPU

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/util/cuda_kernel_helper.h>

// If tensorflow is too old, this does not exist; if tensorflow is too new, it has an incompatible definition
#define CUDA_AXIS_KERNEL_LOOP(i, n, axis)                                  \
  for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n.axis; \
       i += blockDim.axis * gridDim.axis)

using namespace tensorflow;

__global__ void upload_background(cudaSurfaceObject_t dest_surface, TTypes<float, 4>::ConstTensor const src_tensor, int const frames_per_row, dim3 const total_threads)
{
    auto const batch_size = src_tensor.dimension(0);
    auto const frame_height = src_tensor.dimension(1);
    auto const frame_width = src_tensor.dimension(2);
    auto const channels = src_tensor.dimension(3);

    CUDA_AXIS_KERNEL_LOOP(dest_x, total_threads, x) {
        CUDA_AXIS_KERNEL_LOOP(dest_y, total_threads, y) {
            auto const iib = dest_y / frame_height * frames_per_row + dest_x / frame_width;
            if (iib < batch_size) {

                auto const x_in_frame = dest_x % frame_width;
                auto const y_in_frame = frame_height - 1 - dest_y % frame_height;  // the vertical flip ensures that our images are top-row-first, as in tensorflow
                if (channels == 1) {
                    auto const &value = src_tensor(iib, y_in_frame, x_in_frame, 0);
                    surf2Dwrite(float4{value, value, value, 1.f}, dest_surface, dest_x * 16, dest_y);  // *16 is required because surface-writes use byte addressing (!)
                } else if (channels == 3) {
                    surf2Dwrite(float4{
                        src_tensor(iib, y_in_frame, x_in_frame, 0),
                        src_tensor(iib, y_in_frame, x_in_frame, 1),
                        src_tensor(iib, y_in_frame, x_in_frame, 2),
                        1.f,
                    }, dest_surface, dest_x * 16, dest_y);
                }
            }
        }
    }
}

void launch_background_upload(
    cudaArray_t &dest_array, Tensor const &src_tensor,
    int const dest_height, int const dest_width,
    Eigen::GpuDevice const &device
) {
    cudaResourceDesc dest_resource_descriptor;
    dest_resource_descriptor.resType = cudaResourceTypeArray;
    dest_resource_descriptor.res.array.array = dest_array;
    cudaSurfaceObject_t dest_surface;
    if (auto const err = cudaCreateSurfaceObject(&dest_surface, &dest_resource_descriptor))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    auto const config = GetCuda2DLaunchConfig(dest_width, dest_height, device);
    auto const src = src_tensor.tensor<float, 4>();
    upload_background<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
        dest_surface,
        src,
        dest_width / src_tensor.dim_size(2),
        config.virtual_thread_count
    );

    if (auto const err = cudaDestroySurfaceObject(dest_surface))
        LOG(FATAL) << "cudaDestroySurfaceObject failed: " << cudaGetErrorName(err);
}

__global__ void download_pixels(TTypes<float, 4>::Tensor pixels, cudaSurfaceObject_t const src_surface, cudaSurfaceObject_t const src_surface1, cudaSurfaceObject_t const src_surface2, cudaSurfaceObject_t const src_surface3, cudaSurfaceObject_t const src_surface4, cudaSurfaceObject_t const src_surface5, cudaSurfaceObject_t const src_surface6, cudaSurfaceObject_t const src_surface7, int const frames_per_row, dim3 const total_threads)
{
    auto const batch_size = pixels.dimension(0);
    auto const frame_height = pixels.dimension(1);
    auto const frame_width = pixels.dimension(2);
    auto const channels = pixels.dimension(3);

    CUDA_AXIS_KERNEL_LOOP(src_x, total_threads, x) {
        CUDA_AXIS_KERNEL_LOOP(src_y, total_threads, y) {
            auto const iib = src_y / frame_height * frames_per_row + src_x / frame_width;
            if (iib < batch_size) {

                auto const pixel = surf2Dread<float4>(src_surface, src_x * 16, src_y);  // *16 is required because surface-loads use byte addressing (!)
                auto const pixel1 = surf2Dread<float4>(src_surface1, src_x * 16, src_y);  // *16 is required because surface-loads use byte addressing (!)
                auto const pixel2 = surf2Dread<float4>(src_surface2, src_x * 16, src_y);  // *16 is required because surface-loads use byte addressing (!)
                auto const pixel3 = surf2Dread<float4>(src_surface3, src_x * 16, src_y);  // *16 is required because surface-loads use byte addressing (!)
                auto const pixel4 = surf2Dread<float4>(src_surface4, src_x * 16, src_y);  // *16 is required because surface-loads use byte addressing (!)
                auto const pixel5 = surf2Dread<float4>(src_surface5, src_x * 16, src_y);  // *16 is required because surface-loads use byte addressing (!)
                auto const pixel6 = surf2Dread<float4>(src_surface6, src_x * 16, src_y);  // *16 is required because surface-loads use byte addressing (!)
                auto const pixel7 = surf2Dread<float4>(src_surface7, src_x * 16, src_y);  // *16 is required because surface-loads use byte addressing (!)

                auto const x_in_frame = src_x % frame_width;
                auto const y_in_frame = frame_height - 1 - src_y % frame_height;  // the vertical flip ensures that our images are top-row-first, as in tensorflow
                pixels(iib, y_in_frame, x_in_frame, 0) = pixel.x;
                pixels(iib, y_in_frame, x_in_frame, 1) = pixel.y;
                pixels(iib, y_in_frame, x_in_frame, 2) = pixel.z;
                pixels(iib, y_in_frame, x_in_frame, 3) = pixel.w;
                pixels(iib, y_in_frame, x_in_frame, 4) = pixel1.x;
                pixels(iib, y_in_frame, x_in_frame, 5) = pixel1.y;
                pixels(iib, y_in_frame, x_in_frame, 6) = pixel1.z;
                pixels(iib, y_in_frame, x_in_frame, 7) = pixel1.w;
                pixels(iib, y_in_frame, x_in_frame, 8) = pixel2.x;
                pixels(iib, y_in_frame, x_in_frame, 9) = pixel2.y;
                pixels(iib, y_in_frame, x_in_frame, 10) = pixel2.z;
                pixels(iib, y_in_frame, x_in_frame, 11) = pixel2.w;
                pixels(iib, y_in_frame, x_in_frame, 12) = pixel3.x;
                pixels(iib, y_in_frame, x_in_frame, 13) = pixel3.y;
                pixels(iib, y_in_frame, x_in_frame, 14) = pixel3.z;
                pixels(iib, y_in_frame, x_in_frame, 15) = pixel3.w;
                pixels(iib, y_in_frame, x_in_frame, 16) = pixel4.x;
                pixels(iib, y_in_frame, x_in_frame, 17) = pixel4.y;
                pixels(iib, y_in_frame, x_in_frame, 18) = pixel4.z;
                pixels(iib, y_in_frame, x_in_frame, 19) = pixel4.w;
                pixels(iib, y_in_frame, x_in_frame, 20) = pixel5.x;
                pixels(iib, y_in_frame, x_in_frame, 21) = pixel5.y;
                pixels(iib, y_in_frame, x_in_frame, 22) = pixel5.z;
                pixels(iib, y_in_frame, x_in_frame, 23) = pixel5.w;
                pixels(iib, y_in_frame, x_in_frame, 24) = pixel6.x;
                pixels(iib, y_in_frame, x_in_frame, 25) = pixel6.y;
                pixels(iib, y_in_frame, x_in_frame, 26) = pixel6.z;
                pixels(iib, y_in_frame, x_in_frame, 27) = pixel6.w;
                pixels(iib, y_in_frame, x_in_frame, 28) = pixel7.x;
                pixels(iib, y_in_frame, x_in_frame, 29) = pixel7.y;
                pixels(iib, y_in_frame, x_in_frame, 30) = pixel7.z;
                pixels(iib, y_in_frame, x_in_frame, 31) = pixel7.w;
            }
        }
    }
}

void launch_pixels_download(
    Tensor &dest_tensor, cudaArray_t const &src_array, cudaArray_t const &src_array1, cudaArray_t const &src_array2, cudaArray_t const &src_array3, cudaArray_t const &src_array4, cudaArray_t const &src_array5, cudaArray_t const &src_array6, cudaArray_t const &src_array7,
    int const src_height, int const src_width,
    Eigen::GpuDevice const &device
) {
    cudaResourceDesc src_resource_descriptor;
    src_resource_descriptor.resType = cudaResourceTypeArray;
    src_resource_descriptor.res.array.array = src_array;
    cudaSurfaceObject_t src_surface;
    if (auto const err = cudaCreateSurfaceObject(&src_surface, &src_resource_descriptor))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    cudaResourceDesc src_resource_descriptor1;
    src_resource_descriptor1.resType = cudaResourceTypeArray;
    src_resource_descriptor1.res.array.array = src_array1;
    cudaSurfaceObject_t src_surface1;
    if (auto const err = cudaCreateSurfaceObject(&src_surface1, &src_resource_descriptor1))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    cudaResourceDesc src_resource_descriptor2;
    src_resource_descriptor2.resType = cudaResourceTypeArray;
    src_resource_descriptor2.res.array.array = src_array2;
    cudaSurfaceObject_t src_surface2;
    if (auto const err = cudaCreateSurfaceObject(&src_surface2, &src_resource_descriptor2))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    cudaResourceDesc src_resource_descriptor3;
    src_resource_descriptor3.resType = cudaResourceTypeArray;
    src_resource_descriptor3.res.array.array = src_array3;
    cudaSurfaceObject_t src_surface3;
    if (auto const err = cudaCreateSurfaceObject(&src_surface3, &src_resource_descriptor3))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    cudaResourceDesc src_resource_descriptor4;
    src_resource_descriptor4.resType = cudaResourceTypeArray;
    src_resource_descriptor4.res.array.array = src_array4;
    cudaSurfaceObject_t src_surface4;
    if (auto const err = cudaCreateSurfaceObject(&src_surface4, &src_resource_descriptor4))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    cudaResourceDesc src_resource_descriptor5;
    src_resource_descriptor5.resType = cudaResourceTypeArray;
    src_resource_descriptor5.res.array.array = src_array5;
    cudaSurfaceObject_t src_surface5;
    if (auto const err = cudaCreateSurfaceObject(&src_surface5, &src_resource_descriptor5))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    cudaResourceDesc src_resource_descriptor6;
    src_resource_descriptor6.resType = cudaResourceTypeArray;
    src_resource_descriptor6.res.array.array = src_array6;
    cudaSurfaceObject_t src_surface6;
    if (auto const err = cudaCreateSurfaceObject(&src_surface6, &src_resource_descriptor6))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    cudaResourceDesc src_resource_descriptor7;
    src_resource_descriptor7.resType = cudaResourceTypeArray;
    src_resource_descriptor7.res.array.array = src_array7;
    cudaSurfaceObject_t src_surface7;
    if (auto const err = cudaCreateSurfaceObject(&src_surface7, &src_resource_descriptor7))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);


    auto const config = GetCuda2DLaunchConfig(src_width, src_height, device);
    auto dest = dest_tensor.tensor<float, 4>();
    download_pixels<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
        dest,
        src_surface, src_surface1, src_surface2, src_surface3, src_surface4, src_surface5, src_surface6, src_surface7,
        src_width / dest_tensor.dim_size(2),
        config.virtual_thread_count
    );

    if (auto const err = cudaDestroySurfaceObject(src_surface))
        LOG(FATAL) << "cudaDestroySurfaceObject failed: " << cudaGetErrorName(err);
}
