#include "illixr/plugin.hpp"
#include "illixr/opencv_data_types.hpp"
#include "illixr/data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/eye_tracking.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>  
#include <opencv2/imgcodecs.hpp>  
#include <opencv2/core.hpp> 
#include <opencv2/core/mat.hpp>

#include <filesystem>
#include <shared_mutex>

#include "mmio.h"

#define GRAPHICS_STATUS (ptr + 0x00)
#define GRAPHICS_IN     (ptr + 0x04)
#define GRAPHICS_OUT    (ptr + 0x0C)
#define GRAPHICS_DMA    (dma_ptr)

using namespace ILLIXR;

enum EYE_BACKEND {
    CPU,
    GPU,
    NPU
};
static constexpr const int width_ = 160;
static constexpr const int height_ = 240;

class eye_tracking_target_impl : public eye_tracking_target {
public:
    explicit eye_tracking_target_impl(const phonebook* const pb)
        : sb{pb->lookup_impl<switchboard>()}
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

            int backend = 0;
            const char* eye_tracking_env = std::getenv("ILLIXR_EYE_TRACKING");
            if (eye_tracking_env == nullptr) {
                std::cout << "[illixr guest] ILLIXR_EYE_TRACKING not set. Defaulting to CPU." << std::endl;
            } else {
                backend = std::stoi(eye_tracking_env);
                std::cout << "[illixr guest] ILLIXR_EYE_TRACKING: " << eye_tracking_env << std::endl;
            }

            if (backend == 0) {
                eye_tracking_backend = CPU;
                session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                input_tensor_ = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                input_shape_.data(), input_shape_.size()));
                output_tensor_ = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                output_shape_.data(), output_shape_.size()));
                
                // Initialize the lookup table for gamma correction
                lut = cv::Mat(256, 1, CV_8UC1);
                double gamma = 0.8; 
                for (int i = 0; i < 256; ++i) {
                    lut.at<uchar>(i) = cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, gamma));
                }

                clahe = cv::createCLAHE(1.5, cv::Size(8, 8));
            } else if (value == 1) {
                eye_tracking_backend = GPU;
                
                std::cout << "[eye tracking] mapping MMIO" << std::endl;
                int mem_fd;
                mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
                ptr = (intptr_t) mmap(NULL, 16, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0x4000);
            
                std::cout << "[eye tracking] mapping DMA" << std::endl;
                int mem_fd2;
                mem_fd2 = open("/dev/mem", O_RDWR | O_SYNC);
                dma_ptr = (intptr_t) mmap(NULL, 50000000, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd2, 0x88000000);

                std::cout << "[illixr target] finished mapping" << std::endl;

            } else if (value == 2) {
                eye_tracking_backend = NPU;
            } else {
                std::cout << "[illixr guest] Invalid value for ILLIXR_EYE_TRACKING. Defaulting to CPU." << std::endl;
            }
        }




    eye_position_type get_eye_position()  {
        switchboard::ptr<const eye_type> eye_pos = _m_eye_raw.get_ro_nullable();

       if (!eye_pos) {
            return eye_position_type{_m_clock->now(), 0.0, 0.0};
       }

        float pred_x = eye_pos->eye_x_true;
        float pred_y = eye_pos->eye_y_true;

        std::cout << "actual fovea: " << pred_x << ", " << pred_y << std::endl;

        if (eye_tracking_backend == CPU) {
            // preprocess image 
            cv::Mat img = preprocess_img(eye_pos->eye_img);
            const size_t total_elements = width_ * height_;
            if (img.total() != total_elements) {
                throw std::runtime_error("Dimension mismatch between img_scaled and input_image_");
            }
            std::memcpy(input_image_.data(), img.ptr<float>(), total_elements * sizeof(float));

            const char* input_names[] = {"x"};
            const char* output_names[] = {"conv2d_41"};
            session->Run(run_options, input_names, input_tensor_.get(), 1, output_names, output_tensor_.get(), 1);

            get_fovea(pred_x, pred_y);

        } else if (eye_tracking_backend == GPU) {
            cv::Mat img = preprocess_img(eye_pos->eye_img);

            // copy the image to XDMA
            size_t img_size = img.total() * img.elemSize(); 
            std::memcpy((void*) dma_ptr, img.data, img_size);

            // send bridge stream message to host illixr worker to read the image and run the model on host
            tx_packets[0] = make_start_packet(2, 1, img_size);
            send_packets(tx_packets, 1);

            // receive result and stall on target 
            long int delay_ns = read_delay_time();
            std::cout << "[eye tracking] delaying for: " << delay_ns << std::endl;
            std::this_thread::sleep_for(std::chrono::nanoseconds(delay_ns));

        } else if (eye_tracking_backend == NPU) {

        }


        return eye_position_type{_m_clock->now(), pred_x, pred_y};
    }


private:

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
        // std::cout << "predicted fovea: " << fovea_x << ", " << fovea_y << std::endl; 
    }

    uint32_t make_start_packet(int queue_id, int num_packets, int read_dma_bytes) {

        // construct gpu-command-start message {start, queue ID, number of MMIO packets to read, number of DMA bytes to read}
        uint32_t start_stream   = (uint32_t) 0xFF;
        uint32_t queue          = ((uint32_t) queue_id) & 0xFF;  
        uint32_t size           = ((uint32_t) num_packets) & 0xFF;
        uint32_t dma_bytes      = ((uint32_t) read_dma_bytes) & 0xFF;
        start_stream = (start_stream << 24) | (queue << 16) | (size << 8) | (dma_bytes);

        return start_stream;
    }

    void send_packets(uint32_t* packets, int len) {

        // std::cout << "[illixr guest] sending packets: " << std::endl;;
        for (int i = 0; i < len; i++) {
            // std::cout << "   " << packets[i] << std::endl;
            while ((reg_read8(GRAPHICS_STATUS) & 0x2) == 0) ;
            reg_write32(GRAPHICS_IN, packets[i]);
        }
        
        return;
    }

    long int read_delay_time() {
        return 0;
        // look for one packet containing the amount of time to delay in ns
        while ((reg_read8(GRAPHICS_STATUS) & 0x1) == 0) ;
        return (long int) reg_read32(GRAPHICS_OUT);
    }

    const std::shared_ptr<switchboard>                               sb;
    const std::shared_ptr<const RelativeClock>                       _m_clock;

    switchboard::reader<eye_type>                                    _m_eye_raw;
    EYE_BACKEND eye_tracking_backend{CPU}; 
    
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::RunOptions run_options;
    std::unique_ptr<Ort::Value> input_tensor_;
    std::unique_ptr<Ort::Value> output_tensor_;
    std::unique_ptr<Ort::Session> session;

    std::array<int64_t, 4> input_shape_{1, 1, width_, height_};
    std::array<int64_t, 4> output_shape_{1, 4, width_, height_};
    std::array<float, width_ * height_> input_image_{};
    std::array<float, 4 * width_ * height_> results_{};

    cv::Mat lut;
    cv::Ptr<cv::CLAHE> clahe;

    intptr_t ptr, dma_ptr;
    uint32_t tx_packets[50];
    uint32_t rx_packets[50];
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
