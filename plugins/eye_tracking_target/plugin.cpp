#include "illixr/plugin.hpp"
#include "illixr/opencv_data_types.hpp"
#include "illixr/data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/gpu_model.hpp"
#include "illixr/threadloop.hpp"
#include "illixr/switchboard.hpp"
#include "illixr/eye_tracking_target.hpp"
#include "illixr/read_hpm.h"

#include "ritnet.h"
#include <cstdint>
#include <cstdlib>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>  
#include <opencv2/imgcodecs.hpp>  
#include <opencv2/core.hpp> 
#include <opencv2/core/mat.hpp>

#include <filesystem>
#include <shared_mutex>


using namespace ILLIXR;

enum EYE_BACKEND {
    CP,
    GPU,
    NPU
};
static constexpr const int width_ = 240;
static constexpr const int height_ = 160;

class eye_tracking_target_impl : public eye_tracking_target {
    public:
        explicit eye_tracking_target_impl(const phonebook* const pb)
            : sb{pb->lookup_impl<switchboard>()}
            , gpu{pb->lookup_impl<gpu_model>()}
            , _m_clock{pb->lookup_impl<RelativeClock>()}
            , _m_eye_raw{sb->get_reader<eye_type>("eye_raw")} 
            { 
                env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ILLIXR_EyeTracking");
                session_options = Ort::SessionOptions();
                session_options.SetIntraOpNumThreads(1);
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

                std::string model_path = std::getenv("ILLIXR_EYE_MODEL");   
                if (model_path.empty()) {
                    throw std::runtime_error("Model path is not set. Please set the ILLIXR_EYE_MODEL environment variable.");
                }

                int backend = 2;
                const char* eye_tracking_env = std::getenv("ILLIXR_EYE_TRACKING");
                if (eye_tracking_env == nullptr) {
                    std::cout << "[illixr guest] ILLIXR_EYE_TRACKING not set. Defaulting to NPU." << std::endl;
                } else {
                    backend = std::stoi(eye_tracking_env);
                    std::cout << "[illixr guest] ILLIXR_EYE_TRACKING: " << eye_tracking_env << std::endl;
                }

                if (backend == 0) {
                    eye_tracking_backend = CP;
                    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

                    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                    input_tensor_ = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                    input_shape_.data(), input_shape_.size()));
                    output_tensor_ = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                    output_shape_.data(), output_shape_.size()));
                } else if (backend == 1) {
                    eye_tracking_backend = GPU;
                } else if (backend == 2) {
                    eye_tracking_backend = NPU;
                } else {
                    std::cout << "[illixr guest] Invalid value for ILLIXR_EYE_TRACKING. Defaulting to CPU." << std::endl;
                }

                // Initialize the lookup table for gamma correction
                lut = cv::Mat(256, 1, CV_8UC1);
                double gamma = 0.8; 
                for (int i = 0; i < 256; ++i) {
                    lut.at<uchar>(i) = cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, gamma));
                }

                clahe = cv::createCLAHE(1.5, cv::Size(8, 8));
            }


    eye_position_type get_eye_position()  {
        switchboard::ptr<const eye_type> eye_pos = _m_eye_raw.get_ro_nullable();

        read_counters(counters_before);

       if (!eye_pos) {
            // std::cout << "No eye data" << std::endl;
            return eye_position_type{_m_clock->now(), 0.0, 0.0};
       }

        // std::cout << "actual fovea: " << eye_pos->eye_x_true << ", " << eye_pos->eye_y_true << std::endl;

        float pred_x, pred_y;
        if (eye_tracking_backend == CP) {
            // preprocess image 
            cv::Mat img = preprocess_img(eye_pos->eye_img);
            const size_t total_elements = width_ * height_;
            if (img.total() != total_elements) {
                throw std::runtime_error("Dimension mismatch between img_scaled and input_image_");
            }
            // std::memcpy(input_image_.data(), img.ptr<float>(), total_elements * sizeof(float));

            // const char* input_names[] = {"x"};
            // const char* output_names[] = {"conv2d_41"};
            // session->Run(run_options, input_names, input_tensor_.get(), 1, output_names, output_tensor_.get(), 1);

            // get_fovea(pred_x, pred_y);
            // std::cout << "predicted fovea: " << pred_x << ", " << pred_y << std::endl; 

            read_counters(counters_after);
            std::cout << "eye_tracking_target: " << diff_to_string(counters_after, counters_before) << std::endl;

            return eye_position_type{_m_clock->now(), 0.0, 0.0}; // don't do eye tracking for CPU, just return 0

        } else if (eye_tracking_backend == GPU) {
            cv::Mat img = preprocess_img(eye_pos->eye_img);

            // copy the image to XDMA region
            int img_size = img.total() * img.elemSize(); 
            gpu->copy_to_dma(img.data, img_size);

            print_first_5_rows(img);

            // send bridge stream message to host illixr worker to read the image and run the model on host
            uint32_t response_buffer[3];
            gpu->send_gpu_compute_message(2, img_size, 3, response_buffer);
            std::cout << "Host returned eye pos: " << response_buffer[1] << " " << response_buffer[2] << std::endl;

        } else if (eye_tracking_backend == NPU) {
            cv::Mat img = preprocess_img(eye_pos->eye_img);
            img.convertTo(img, CV_8U);
            int img_size = img.total() * img.elemSize(); 
            gpu->copy_to_dma(img.data, img_size);

            // get [1][160][240][1] c array from the cv::Mat
            // int8_t (*input_image)[160][240][1] = reinterpret_cast<int8_t (*)[160][240][1]>(img.data);

            std::cout << "Gemmini inference" << std::endl;
            // call gemmini function to perform inference
            // gemmini_inference(input_image, pred_x, pred_y);

            std::system("/root/ILLIXR/plugins/eye_tracking_target/gemmini/ritnet-linux");

            std::cout << "finished inference" << std::endl;

        }

        read_counters(counters_after);
        std::cout << diff_to_string(counters_after, counters_before) << std::endl;

        return eye_position_type{_m_clock->now(), eye_pos->eye_x_true, eye_pos->eye_y_true};
    }


private:

    void print_first_5_rows(const cv::Mat& mat) {
        int rows_to_print = std::min(5, mat.rows);
        cv::Mat first_rows = mat(cv::Range(0, rows_to_print), cv::Range::all());

        std::cout << "First " << rows_to_print << " rows of matrix:\n" << first_rows << std::endl;
    }

    cv::Mat preprocess_img(cv::Mat img) {
         // preprocess the eye image for RITnet

         // Gamma correction
         cv::Mat img_lut;
         cv::LUT(img, lut, img_lut);

         // CLAHE
         cv::Mat img_clahe;
         clahe->apply(img_lut, img_clahe);

         // convert to float and normalize
         cv::Mat img_float;
         img_clahe.convertTo(img_float, CV_32F, 1.0 / 255.0);
         img_float = (img_float - 0.5f) / 0.5f;

         // scale and quantize to int8
         double maxVal;
         cv::minMaxLoc(cv::abs(img_float), nullptr, &maxVal);
         float scale = static_cast<float>(maxVal) / 127.0f;
         cv::Mat img_scaled;
         img_float = img_float / scale;
         img_float.convertTo(img_scaled, CV_8S);
         img_scaled.convertTo(img_scaled, CV_32F);

        if (!img_scaled.isContinuous()) {
            img_scaled = img_scaled.clone();
        }

        return img_scaled;
    }

    void get_fovea(float& fovea_x, float& fovea_y) {
        std::vector<uint8_t> argmax_map(height_ * width_);
        float* output_data = output_tensor_->GetTensorMutableData<float>();

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                int max_class = 0;
                float max_value = output_data[0 * height_ * width_ + y * width_ + x];
                for (int c = 1; c < 4; ++c) {
                    float value = output_data[c * height_ * width_ + y * width_ + x];
                    if (value > max_value) {
                        max_value = value;
                        max_class = c;
                    }
                }
                argmax_map[y * width_ + x] = static_cast<uint8_t>(max_class);
            }
        }

        double sum_x = 0.0;
        double sum_y = 0.0;
        int count = 0;

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                if (argmax_map[y * width_ + x] != 0) { // Assuming class 0 is background
                    sum_x += x;
                    sum_y += y;
                    ++count;
                }
            }
        }

        fovea_x = (count > 0) ? (sum_x / count) : 0.0;
        fovea_y = (count > 0) ? (sum_y / count) : 0.0;
    }


    const std::shared_ptr<switchboard>                               sb;
    const std::shared_ptr<const RelativeClock>                       _m_clock;
    const std::shared_ptr<gpu_model>                                 gpu;

    switchboard::reader<eye_type>                                    _m_eye_raw;
    EYE_BACKEND eye_tracking_backend{CP}; 
    
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::RunOptions run_options;
    std::unique_ptr<Ort::Value> input_tensor_;
    std::unique_ptr<Ort::Value> output_tensor_;
    std::unique_ptr<Ort::Session> session;

    std::array<int64_t, 4> input_shape_{1, 1, height_, width_};
    std::array<int64_t, 4> output_shape_{1, 4, height_, width_};
    std::array<float, width_ * height_> input_image_{};
    std::array<float, 4 * width_ * height_> results_{};

    cv::Mat lut;
    cv::Ptr<cv::CLAHE> clahe;

    double counters_before[29] = {0.0};
    double counters_after[29] = {0.0};
};


class eye_tracking_target_plugin : public plugin {
    public:
        eye_tracking_target_plugin(const std::string& name, phonebook* pb)
            : plugin{name, pb} {
            pb->register_impl<eye_tracking_target>(
                std::static_pointer_cast<eye_tracking_target>(std::make_shared<eye_tracking_target_impl>(pb)));
        }
    };
    
PLUGIN_MAIN(eye_tracking_target_plugin);
