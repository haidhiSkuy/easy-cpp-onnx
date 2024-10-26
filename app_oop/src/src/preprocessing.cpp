#include "inferencer.h"
#include <opencv2/opencv.hpp>  
#include <string>

cv::Mat ImagenetOnnx::preprocessing(std::string inputImage)
{ 
    cv::Mat img = cv::imread(inputImage, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        std::abort();
    } 

    cv::Mat preprocessedImage;
    cv::dnn::blobFromImage(img, preprocessedImage, 1 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false); 

    return preprocessedImage;
}