#include "inferencer.h"
#include <string>
#include <opencv2/opencv.hpp>  

int main()
{ 
    std::string modelPath = "/workspaces/ONNX-cpp/onnx/mobilenetv2-12.onnx"; 
    std::string labelPath = "/workspaces/ONNX-cpp/onnx/synset.txt";
    
    ImagenetOnnx inferencer(modelPath, labelPath); 

    std::string imagePath = "/workspaces/ONNX-cpp/sample3.jpg"; 
    cv::Mat preprocessed = inferencer.preprocessing(imagePath);

    inferencer.run(preprocessed);

    return 0;
} 