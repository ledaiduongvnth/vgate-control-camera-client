#include "cudaUtility.h"


// gpuPreImageNetRGB
__global__ void gpuPreImageNetRGB( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const float4 px  = input[ dy * iWidth + dx ];
    const float3 bgr = make_float3(px.x, px.y, px.z);

    output[n * 0 + m] = bgr.x;
    output[n * 1 + m] = bgr.y;
    output[n * 2 + m] = bgr.z;
}


// cudaPreImageNetRGB
cudaError_t cudaPreImageNetRGB( float4* input, size_t inputWidth, size_t inputHeight,
                                float* output, size_t outputWidth, size_t outputHeight,
                                cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPreImageNetRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

    return CUDA(cudaGetLastError());
}