#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp> 
#include <vector> 
#include <numeric> 
#include <cmath>  
#include <fstream>


std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> expVals(logits.size());
    float maxLogit = *std::max_element(logits.begin(), logits.end());

    // Compute exponentials of logits
    float sumExp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        expVals[i] = std::exp(logits[i] - maxLogit);  // Numerical stability improvement
        sumExp += expVals[i];
    }

    // Normalize to get probabilities
    for (size_t i = 0; i < expVals.size(); ++i) {
        expVals[i] /= sumExp;
    }

    return expVals;
}



int main() { 
    std::ifstream file("/workspaces/ONNX-cpp/onnx/synset.txt");
    std::vector<std::string> labels;
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file";
    }


    // Initialize ONNX Runtime environment and session
    Ort::Env mEnv(ORT_LOGGING_LEVEL_WARNING);



    Ort::SessionOptions sessionOptions;

    std::string modelPath = "/workspaces/ONNX-cpp/onnx/mobilenetv2-12.onnx";  
    Ort::Session mSession(mEnv, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr inputName = mSession.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr outputName = mSession.GetOutputNameAllocated(0, allocator);

    // Get input and output dimensions
    size_t numInputNodes = mSession.GetInputCount();
    Ort::TypeInfo inputTypeInfo = mSession.GetInputTypeInfo(0);
    Ort::ConstTensorTypeAndShapeInfo inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> mInputDims = inputTensorInfo.GetShape();  


    int inputSize = 1; 
    for (auto& dim : mInputDims) {
        if (dim == -1) {
            dim = 1; // Setting batch size to 1 if dynamic
        }
        inputSize *= dim; 
    } 
 
    size_t numOutputNodes = mSession.GetOutputCount();
    Ort::TypeInfo outputTypeInfo = mSession.GetOutputTypeInfo(0);
    Ort::ConstTensorTypeAndShapeInfo outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> mOutputDims = outputTensorInfo.GetShape();

    int outputSize = 1;
    for (auto& dim : mOutputDims) {
        if (dim == -1) {
            dim = 1; // Setting batch size to 1 if dynamic
        }
        outputSize *= dim;
    } 

    // Load an input image
    cv::Mat imageArr = cv::imread("/workspaces/ONNX-cpp/sample/sample.jpg");
    if (imageArr.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }
    
    cv::Mat RgbImage;
    cv::cvtColor(imageArr, RgbImage, cv::COLOR_BGR2RGB);

    // Preprocess the image (Resize, normalize, convert to blob format)
    cv::Mat preprocessedImage;
    cv::dnn::blobFromImage(RgbImage, preprocessedImage, 1 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);

    // Check if dimensions match
    size_t inputTensorSize = inputSize;
    if (inputTensorSize != preprocessedImage.total()) {
        std::cout << inputTensorSize << " " << preprocessedImage.total() << std::endl; 
        std::cerr << "Preprocessed image size does not match input tensor size" << std::endl;
        return -1;
    }

    std::vector<float> inputTensorValues(preprocessedImage.begin<float>(), preprocessedImage.end<float>()); 


    // Create input tensors of ORT::Value, which is a tensor format used by ONNX Runtime
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, mInputDims.data(), mInputDims.size()
        )
    ); 

    // Create output tensor buffer
    size_t outputTensorSize = outputSize;
    std::vector<float> outputTensorValues(outputTensorSize);

    // Create output tensors of ORT::Value
    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(),
                                outputTensorSize, mOutputDims.data(), mOutputDims.size()));

    // Input and output names
    std::vector<const char*> inputNames = { inputName.get() };
    std::vector<const char*> outputNames = { outputName.get() };


    // Run the model
    mSession.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1,
                  outputNames.data(), outputTensors.data(), 1); 

    std::vector<float> probabilities = softmax(outputTensorValues);
    
    // Find the class with the highest probability
    auto maxIt = std::max_element(probabilities.begin(), probabilities.end());
    int predictedClass = std::distance(probabilities.begin(), maxIt);
    float maxProb = *maxIt;

    // Output the results
    std::cout << "Predicted class: " << labels[predictedClass] << std::endl;
    std::cout << "Probability: " << maxProb << std::endl;

    return 0;
}
