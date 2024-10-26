#include "inferencer.h" 
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp> 
#include <numeric> 
#include <cmath>  
#include <memory>

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


void ImagenetOnnx::run(cv::Mat preprocessedImage)
{   
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<float> inputTensorValues(preprocessedImage.begin<float>(), preprocessedImage.end<float>());

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                inputTensorSize, mInputDims.data(), mInputDims.size()));

    // Create output tensor buffer
    std::vector<float> outputTensorValues(outputTensorSize);

    // Create output tensors of ORT::Value
    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValues.data(), outputTensorSize, mOutputDims.data(), mOutputDims.size()
        )
    );

    // Input and output names
    std::vector<const char*> inputNames = {"input"};
    std::vector<const char*> outputNames = {"output"};

    // Run the model
    mSession->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1,
                  outputNames.data(), outputTensors.data(), 1); 
    

    std::vector<float> probabilities = softmax(outputTensorValues);

    // Find the class with the highest probability
    auto maxIt = std::max_element(probabilities.begin(), probabilities.end());
    int predictedClass = std::distance(probabilities.begin(), maxIt);
    float maxProb = *maxIt;

    // Output the results
    std::cout << "Predicted class: " << labels[predictedClass] << std::endl;
    std::cout << "Probability: " << maxProb << std::endl;
    
}