#include <onnxruntime_cxx_api.h>
#include <vector>
#include <opencv2/opencv.hpp>  

class ImagenetOnnx { 
    public: 
        ImagenetOnnx(std::string onnxPath, std::string labelPath); 
        cv::Mat preprocessing(std::string inputImage);  
        void run(cv::Mat preprocessedImage);

    private: 
        // get label list
        std::vector<std::string> labels;
        
        // preparing ort session
        std::shared_ptr<Ort::Env> mEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "warning");  
        Ort::SessionOptions sessionOptions; 
        std::shared_ptr<Ort::Session> mSession;

        // input output 
        std::vector<int64_t> mInputDims;
        std::vector<int64_t> mOutputDims;
        
        size_t inputTensorSize;
        size_t outputTensorSize;

};