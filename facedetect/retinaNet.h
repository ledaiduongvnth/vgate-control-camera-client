#include "tensorNet.h"
#include "postProcessRetina.h"


class retinaNet : public tensorNet
{
public:
	float* cudaInput;
    retinaNet();
    ~retinaNet();
    int Detect(float3* imgrgb32, uint32_t width, uint32_t height,
               postProcessRetina* rf, std::vector<FaceDetectInfo>& faceInfo, float threshold);
};


