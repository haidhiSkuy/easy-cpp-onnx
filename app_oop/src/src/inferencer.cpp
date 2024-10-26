#include "inferencer.h"
#include <fstream>
#include <onnxruntime_cxx_api.h>


ImagenetOnnx::ImagenetOnnx(std::string onnxPath, std::string labelPath)
{ 
    std::ifstream file(labelPath);
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file";
    }

    mSession = std::make_shared<Ort::Session>(*mEnv, onnxPath.c_str(), sessionOptions); 
    
    //input output
    size_t numInputNodes = mSession->GetInputCount();
    Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    mInputDims = inputTensorInfo.GetShape();  

    int inputSize = 1; 
    for (auto& dim : mInputDims) {
        if (dim == -1) {
            dim = 1; // Setting batch size to 1 if dynamic
        }
        inputSize *= dim;
    }
    inputTensorSize = inputSize;

    size_t numOutputNodes = mSession->GetOutputCount();
    Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    mOutputDims = outputTensorInfo.GetShape();

    int outputSize = 1;
    for (auto& dim : mOutputDims) {
        if (dim == -1) {
            dim = 1; // Setting batch size to 1 if dynamic
        }
        outputSize *= dim;
    }
    outputTensorSize = outputSize; 

    std::cout << "Load model success" << std::endl;

}