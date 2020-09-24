#include "cudaUtility.h"
#include <npp.h>


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

void imagePadding32f4C(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight, int top, int left)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;

    int nSrcStep = srcWidth * 4 * sizeof(float);

    int nDstStep = dstWidth * 4 * sizeof(float);

    NppiSize oDstSize;
    oDstSize.width = dstWidth;
    oDstSize.height = dstHeight;

    Npp32f aValue[4];
    aValue[0] = 0;
    aValue[1] = 0;
    aValue[2] = 0;
    aValue[3] = 255;


    NppStatus ret = nppiCopyConstBorder_32f_C4R((const Npp32f *)src, nSrcStep, oSrcSize,
                                                (Npp32f *)dst, nDstStep, oDstSize, top, left, aValue);
    if(ret != NPP_SUCCESS) {
        printf("imageResize_32f_C4R failed %d.\n", ret);
        throw std::exception();
    }
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