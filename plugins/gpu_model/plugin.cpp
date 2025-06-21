#include "illixr/plugin.hpp"

#include "illixr/data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/gpu_model.hpp"
#include "illixr/read_hpm.h"

#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <shared_mutex>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "include/mmio.h"

#define GRAPHICS_STATUS (ptr + 0x00)
#define GRAPHICS_IN     (ptr + 0x04)
#define GRAPHICS_OUT    (ptr + 0x0C)
#define GRAPHICS_DMA    (dma_ptr)

using namespace ILLIXR;

class gpu_model_impl : public gpu_model {
public:
    explicit gpu_model_impl(const phonebook* const pb)
        : sb{pb->lookup_impl<switchboard>()}
        , _m_clock{pb->lookup_impl<RelativeClock>()}
    {

        std::cout << "[illixr target] mapping MMIO" << std::endl;
        int mem_fd;
        mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
        ptr = (intptr_t) mmap(NULL, 16, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0x4000);
      
        std::cout << "[illixr target] mapping DMA" << std::endl;
        int mem_fd2;
        mem_fd2 = open("/dev/mem", O_RDWR | O_SYNC);
        dma_ptr = (intptr_t) mmap(NULL, 50000000, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd2, 0x88000000);

        std::cout << "[illixr target] finished mapping" << std::endl;

    }

    void send_gpu_render_message(fast_pose_type current_pose, eye_position_type eye_pos, int queue_id, int read_dma_bytes, int num_response_expected, uint32_t* response_buffer) {
        read_counters(counters_before);

        // std::cout << "[illixr guest] sending gpu message" << std::endl;

        tx_packets[0] = make_start_packet(queue_id, 10, read_dma_bytes);
        make_pose_packets(tx_packets, current_pose.pose, 1);
        make_eye_pose_packets(tx_packets, eye_pos, 8);

        tx_packets[9] = float_to_uint32(shading_rate); // shading rate packet

        // send to bridge
        // bridge will pause target execution while render is occurring
        send_packets(tx_packets, 11); // start packet, 7 pose packets, 2 eye packets, 1 shading rate packet

        for (int i = 0; i < num_response_expected; i++) {
            response_buffer[i] = read_packet();
            // std::cout << " gpu model received: " << response_buffer[i] << std::endl;
        }

        // std::cout << "[illixr guest] render delaying for: " << response_buffer[0] << std::endl;
        std::this_thread::sleep_for(std::chrono::nanoseconds(response_buffer[0]));

        // based on the delay time, we can update the shading rate
        if (response_buffer[0] < 1.23) {
            std::cout << "[gpu_model] delay time: " << response_buffer[0] / 1e6 << "ms, setting shading rate to 0" << std::endl;
            shading_rate = 0; 
        } else if (response_buffer[0] < 1.26) {
            std::cout << "[gpu_model] delay time: " << response_buffer[0] / 1e6 << "ms, setting shading rate to 1" << std::endl;
            shading_rate = 1; 
        } else {
            std::cout << "[gpu_model] delay time: " << response_buffer[0] / 1e6 << "ms, setting shading rate to 2" << std::endl;
            shading_rate = 2; 
        }

        read_counters(counters_after);
        std::cout << "gpu_model: " << diff_to_string(counters_after, counters_before) << std::endl;

    }

    void send_gpu_compute_message(int queue_id, int read_dma_bytes, int num_response_expected, uint32_t* response_buffer) {
        tx_packets[0] = make_start_packet(queue_id, 1, read_dma_bytes);
        tx_packets[1] = 0x00000000; // dummy packet because bridge driver needs at least 2 packets

        // std::cout << "[illixr guest] sending compute message: " << queue_id << std::endl;
        send_packets(tx_packets, 2);

        for (int i = 0; i < num_response_expected; i++) {
            response_buffer[i] = read_packet();
            // std::cout << " gpu model received: " << response_buffer[i] << std::endl;
        }

        // std::cout << "[illixr guest] compute delaying for: " << response_buffer[0] << std::endl;
        std::this_thread::sleep_for(std::chrono::nanoseconds(response_buffer[0]));
    }

    void copy_to_dma(void* data, int bytes) {
        std::unique_lock lock{dma_mutex};
        std::memcpy((void*) dma_ptr, data, bytes);

        
    }

    void copy_from_dma(void* data, int bytes) {
        std::unique_lock lock{dma_mutex};
        std::memcpy((void*) data, (void*) dma_ptr, bytes);
    }


private:

    double counters_before[29] = {0.0};
    double counters_after[29] = {0.0};

    uint32_t make_start_packet(int queue_id, int num_packets, int read_dma_bytes) {

        // construct gpu-command-start message {start, queue ID, number of MMIO packets to read, number of DMA bytes to read}
        uint32_t start_stream   = (uint32_t) 0xFF;
        uint32_t queue          = ((uint32_t) queue_id) & 0xFF;  
        uint32_t size           = ((uint32_t) num_packets) & 0xFF;
        uint32_t dma_bytes      = ((uint32_t) read_dma_bytes) & 0xFF;
        start_stream = (start_stream << 24) | (queue << 16) | (size << 8) | (dma_bytes);

        return start_stream;
    }

    void make_pose_packets(uint32_t* packets, pose_type pose, int start_index) {
        packets[start_index] = float_to_uint32(pose.position.x());
        packets[start_index+1] = float_to_uint32(pose.position.y());
        packets[start_index+2] = float_to_uint32(pose.position.z());
        packets[start_index+3] = float_to_uint32(pose.orientation.w());
        packets[start_index+4] = float_to_uint32(pose.orientation.x());
        packets[start_index+5] = float_to_uint32(pose.orientation.y());
        packets[start_index+6] = float_to_uint32(pose.orientation.z());
    } 

    void make_eye_pose_packets(uint32_t* packets, eye_position_type eye_pos, int start_index) {
        packets[start_index] = float_to_uint32(eye_pos.eye_x);
        packets[start_index+1] = float_to_uint32(eye_pos.eye_y);
    }

    long int read_packet() {
        // return 0;
        while ((reg_read8(GRAPHICS_STATUS) & 0x1) == 0) ;
        return (long int) reg_read32(GRAPHICS_OUT);
    }

    void send_packets(uint32_t* packets, int len) {
        // return;
        std::unique_lock lock{bridge_mutex};

        std::cout << "[illixr guest] sending packets: " << std::endl;;
        for (int i = 0; i < len; i++) {
            std::cout << "   " << packets[i] << std::endl;
            while ((reg_read8(GRAPHICS_STATUS) & 0x2) == 0) ;
            reg_write32(GRAPHICS_IN, packets[i]);
        }
        std::cout << "[illixr guest] finished sending packets" << std::endl;
    }

    uint32_t float_to_uint32(float val) {
        uint32_t result;
        memcpy(&result, &val, sizeof(float));
        return result;
    }


    mutable std::atomic<bool>                                        first_time{true};
    const std::shared_ptr<switchboard>                               sb;
    const std::shared_ptr<const RelativeClock>                       _m_clock;

    mutable std::shared_mutex                                        bridge_mutex;
    mutable std::shared_mutex                                        dma_mutex;

    intptr_t ptr, dma_ptr;

    uint32_t tx_packets[50];
    uint32_t rx_packets[50];

    int shading_rate = 0;

};

class gpu_model_plugin : public plugin {
public:
    gpu_model_plugin(const std::string& name, phonebook* pb)
        : plugin{name, pb} {
        pb->register_impl<gpu_model>(
            std::static_pointer_cast<gpu_model>(std::make_shared<gpu_model_impl>(pb)));
    }
};

PLUGIN_MAIN(gpu_model_plugin);
