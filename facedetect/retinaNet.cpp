/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gstCamera.h>
#include "retinaNet.h"
#include "cudaUtility.h"


cudaError_t cudaPreImageNetRGB( float4* input, size_t inputWidth, size_t inputHeight,
                                float* output, size_t outputWidth, size_t outputHeight,
                                cudaStream_t stream );

void imagePadding32f4C(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight, int top, int left);


// constructor
retinaNet::retinaNet(int cameraWidth, int cameraHeight, std::string camera_source)
{
    gstCamera *camera = gstCamera::Create(cameraWidth, cameraHeight, camera_source.c_str());
    if( !camera )
    {
        printf("failed to initialize camera device\n");
        throw std::exception();
    }
    if( !camera->Open() )
    {
        printf("failed to open camera for streaming\n");
        throw std::exception();
    }
    camera->CaptureRGBA(&cudaInput, 1000, true);
    camera->Close();
}


// Destructor
retinaNet::~retinaNet()
{

}


// Create
retinaNet* retinaNet::Create(int cameraWidth, int cameraHeight, std::string camera_source)
{
	retinaNet* net = new retinaNet(cameraWidth, cameraHeight, camera_source);

	const char* model_path  = "../facedetect/model/retina.onnx";
	const char* input_blob  = "data_input";
    std::vector<std::string> output_blobs;
    output_blobs = {
            "face_rpn_bbox_pred_stride32_Y",
            "face_rpn_landmark_pred_stride32_Y",
            "face_rpn_cls_prob_reshape_stride32_Reshape_Y",
            "face_rpn_bbox_pred_stride16_Y",
            "face_rpn_landmark_pred_stride16_Y",
            "face_rpn_cls_prob_reshape_stride16_Reshape_Y",
            "face_rpn_bbox_pred_stride8_Y",
            "face_rpn_landmark_pred_stride8_Y",
            "face_rpn_cls_prob_reshape_stride8_Reshape_Y"
    };


    const uint32_t maxBatchSize = 1;

	if( !net->LoadNetwork(NULL, model_path, NULL, input_blob, output_blobs, maxBatchSize) )
	{
		printf(LOG_TRT "failed to load retinaNet model\n");
		return NULL;
	}

	return net;
}

int retinaNet::Detect(float* rgba, uint32_t width, uint32_t height, postProcessRetina* rf, std::vector<FaceDetectInfo>& faceInfo, float threshold)
{

    if( !rgba || width == 0 || height == 0  )
    {
        printf(LOG_TRT "detectNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
        return -1;
    }

    imagePadding32f4C(rgba, width, height, cudaInput, width, width, 0, 0);

    if( CUDA_FAILED(cudaPreImageNetRGB((float4*)cudaInput, width, width, mInputCUDA, 640, 640, GetStream())) )
    {
        printf(LOG_TRT "imageNet::PreProcess() -- cudaPreImageNetNormMeanRGB() failed\n");
        return false;
    }

    void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA, mOutputs[1].CUDA, mOutputs[2].CUDA, mOutputs[3].CUDA, mOutputs[4].CUDA,
                                 mOutputs[5].CUDA, mOutputs[6].CUDA, mOutputs[7].CUDA, mOutputs[8].CUDA};


    if( !mContext->execute(1, inferenceBuffers) )
    {
        printf(LOG_TRT "detectNet::Detect() -- failed to execute TensorRT context\n");
        return -1;
    }

    CUDA(cudaDeviceSynchronize());

    int numDetections = 0;
    std::vector<std::vector<float>> results;

    for (int i = 0; i < 9; i++) {
        std::vector<float> outputi = std::vector<float>(mOutputs[i].CPU, mOutputs[i].CPU + mOutputs[i].size / 4);
        results.emplace_back(outputi);
    }
    rf->detect(results, threshold, faceInfo, 640);

    return numDetections;
}




