#include "trtnetbase.h"
#include "trtutility.h"
#include <assert.h>
#include <iterator>
#include <memory>
#include "NvOnnxParser.h"

using namespace std;

#ifdef USE_TENSORRT_INT8
Int8EntropyCalibrator2::Int8EntropyCalibrator2()
{
    mReadCache = true;
    calibrationTableName = std::string("../model/mnet-deconv-0517.table.int8");
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{

}

bool Int8EntropyCalibrator2::checkCalibrationTable()
{
    std::ifstream input(calibrationTableName, std::ios::binary);
    if(!input.good()){
        return false;
    }
    input.close();
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length)
{
    mCalibrationCache.clear();
    std::ifstream input(calibrationTableName, std::ios::binary);

    input >> std::noskipws;
    if (mReadCache && input.good()){
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
    }
    length = mCalibrationCache.size();
    input.close();

    return length ? &mCalibrationCache[0] : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length)
{
    std::ofstream output(calibrationTableName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}
#endif // USE_TENSORRT_INT8

//This function is used to trim space
string TrtNetBase::stringtrim(string s)
{
    int i = 0;
    while (s[i] == ' ') {
        i++;
    }
    s = s.substr(i);
    i = s.size()-1;
    while (s[i] == ' ') {
        i--;
    }

    s = s.substr(0, i + 1);
    return s;
}

uint32_t TrtNetBase::getBatchSize() const
{
    return batchSize;
}

uint32_t TrtNetBase::getMaxBatchSize() const
{
    return maxBatchSize;
}

int TrtNetBase::getNetWidth() const
{
    return netWidth;
}

int TrtNetBase::getNetHeight() const
{
    return netHeight;
}

int TrtNetBase::getChannel() const
{
    return channel;
}

void *&TrtNetBase::getBuffer(const int &index)
{
    assert(index >= 0 && index < numBinding);
    return buffers[index];
}

float *&TrtNetBase::getInputBuf()
{
    return inputBuffer;
}

void TrtNetBase::setForcedFp32(const bool &forcedFp32)
{
    useFp32 = forcedFp32;
}

void TrtNetBase::setDumpResult(const bool &dumpResult)
{
    this->dumpResult = dumpResult;
}

void TrtNetBase::setTrtProfilerEnabled(const bool &enableTrtProfiler)
{
    this->enableTrtProfiler = enableTrtProfiler;
}

TrtNetBase::TrtNetBase(string netWorkName)
{
    pLogger = new Logger();
    profiler = new Profiler();
    runtime = NULL;
    engine = NULL;
    context = NULL;

    batchSize = 0;
    channel = 0;
    netWidth = 0;
    netHeight = 0;

    useFp32 = false;

    dumpResult = false;
    resultFile = "result.txt";
    enableTrtProfiler = false;
    this->netWorkName = netWorkName;
}

TrtNetBase::~TrtNetBase()
{
    delete pLogger;
    delete profiler;
}

bool TrtNetBase::parseNet(const string& deployfile)
{
    ifstream readfile;
    string line;
    readfile.open(deployfile, ios::in);
    if (!readfile) {
        printf("the deployfile doesn't exist!\n");
        return false;
    }

    while (1) {
        getline(readfile, line);
        string::size_type index;

        index = line.find("input_param");
        if (index == std::string::npos) {
            continue;
        }

        getline(readfile, line);

        index = line.find("dim:", 0);

        string first = line.substr(index + 5);
        string second = line.substr(index + 12);
        string third = line.substr(index + 19);
        string fourth = line.substr(index + 28);

        batchSize = atoi(stringtrim(first).c_str());
        assert(batchSize > 0);

        channel = atoi(stringtrim(second).c_str());
        assert(channel > 0);

        netHeight = atoi(stringtrim(third).c_str());
        assert(netHeight > 0);

        netWidth = atoi(stringtrim(fourth).c_str());
        assert(netWidth > 0);

        break;
    }

    printf("batchSize:%d, channel:%d, netHeight:%d, netWidth:%d.\n", batchSize, channel, netHeight, netWidth);

    readfile.close();

    return true;
}

void TrtNetBase::buildTrtContext(const std::string& modelfile, bool bUseCPUBuf)
{
    caffeToTRTModel(modelfile, NULL);
    runtime = createInferRuntime(*pLogger);
    engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), NULL);
    trtModelStream->destroy();
    context = engine->createExecutionContext();
    context->setProfiler(profiler);
    allocateMemory(bUseCPUBuf);
}

void TrtNetBase::destroyTrtContext(bool bUseCPUBuf)
{
    releaseMemory(bUseCPUBuf);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void TrtNetBase::caffeToTRTModel(const std::string& modelFile, nvcaffeparser1::IPluginFactory* pluginFactory){
    // create API root class - must span the lifetime of the engine usage
    IBuilder* builder = createInferBuilder(*pLogger);
    INetworkDefinition* network = builder->createNetwork();
    Logger gLogger;
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
    std::ifstream onnx_file(modelFile, std::ios::binary | std::ios::ate);
    std::streamsize file_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
    std::vector<char> onnx_buf(file_size);
    onnx_file.read(onnx_buf.data(), onnx_buf.size());
    if (!parser->parse(onnx_buf.data(), onnx_buf.size())) {
        int nerror = parser->getNbErrors();
        for (int i = 0; i < nerror; ++i) {
            nvonnxparser::IParserError const *error = parser->getError(i);
            std::cerr << "ERROR: "
                      << error->file() << ":" << error->line()
                      << " In function " << error->func() << ":\n"
                      << "[" << static_cast<int>(error->code()) << "] " << error->desc()
                      << std::endl;
        }
    }
    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildCudaEngine(*network);

    assert(engine);
    network->destroy();
    parser->destroy();
    trtModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
}

