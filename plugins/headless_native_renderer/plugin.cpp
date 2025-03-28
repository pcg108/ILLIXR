#include <array>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "illixr/global_module_defs.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/pose_prediction.hpp"
#include "illixr/switchboard.hpp"
#include "illixr/threadloop.hpp"
#include "illixr/vk_util/headless_sink.hpp"
#include "illixr/vk_util/render_pass.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "illixr/gl_util/lib/tiny_obj_loader.h"

#include "mmio.h"

#define GRAPHICS_STATUS (ptr + 0x00)
#define GRAPHICS_IN     (ptr + 0x04)
#define GRAPHICS_OUT    (ptr + 0x0C)
#define GRAPHICS_DMA    (dma_ptr)

using namespace ILLIXR;

const record_header mtp_record{"mtp_record",
    {
        {"render_pose_c", typeid(std::size_t)},
        {"render_c", typeid(std::size_t)},
        {"tw_pose_c", typeid(std::size_t)},
        {"timewarp_c", typeid(std::size_t)},
        {"MTP_ns", typeid(std::chrono::nanoseconds)},
    }};

class native_renderer : public threadloop {
public:
    native_renderer(const std::string& name_, phonebook* pb)
        : threadloop{name_, pb}
        , sb{pb->lookup_impl<switchboard>()}
        , pp{pb->lookup_impl<pose_prediction>()}
        , _m_clock{pb->lookup_impl<RelativeClock>()}
        , last_fps_update{std::chrono::duration<long, std::nano>{0}}
        , mtp_logger{record_logger_} {
        spdlogger(std::getenv("NATIVE_RENDERER_LOG_LEVEL"));
    }

    /**
     * @brief Sets up the thread for the plugin.
     *
     * This function initializes depth images, offscreen targets, command buffers, sync objects,
     * application and timewarp passes, offscreen and swapchain framebuffers. Then, it initializes
     * application and timewarp with their respective passes.
     */
    void _p_thread_setup() override {

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

    /**
     * @brief Executes one iteration of the plugin's main loop.
     */
    void _p_one_iteration() override {

        uint64_t before_render_pose = rdcycle();

            // offload render 
            auto render_pose = pp->get_fast_pose();

        uint64_t after_render_pose = rdcycle();

        tx_packets[0] = make_start_packet(0, 7, 0);
        make_pose_packets(tx_packets, render_pose.pose);


        uint64_t before_render = rdcycle();

            // send to bridge
            // bridge will pause target execution while render is occurring
            send_packets(tx_packets, 8);

            // get the amount of time to stall from the bridge
            // block to simulate target execution
            long int delay_ns = read_delay_time();
            std::cout << "[illixr guest] delaying for: " << delay_ns << std::endl;
            std::this_thread::sleep_for(std::chrono::nanoseconds(900000000));

        uint64_t after_render = rdcycle();

        uint64_t before_tw_pose = rdcycle();

            // offload timewarp 
            auto timewarp_pose = pp->get_fast_pose().pose;

        uint64_t after_tw_pose = rdcycle();

        tx_packets[0] = make_start_packet(1, 7, 0);
        make_pose_packets(tx_packets, timewarp_pose);

        uint64_t before_tw = rdcycle();

            send_packets(tx_packets, 8);

            delay_ns = read_delay_time();
            std::cout << "[illixr guest] delaying for: " << delay_ns << std::endl;
            std::this_thread::sleep_for(std::chrono::nanoseconds(delay_ns));

        uint64_t after_tw = rdcycle();

        std::cout << "MTP: " << duration2double<std::milli>(_m_clock->now() - timewarp_pose.pose.sensor_time) << std::endl;

        mtp_logger.log(record{mtp_record,
            {
                {(size_t) (after_render_pose - before_render_pose)},
                {(size_t) (after_render - before_render)},
                {(size_t) (after_tw_pose - before_tw_pose)},
                {(size_t) (after_tw - before_tw)},
                {_m_clock->now() - render_pose.pose.sensor_time},
            }});

    }

private:

    uint32_t make_start_packet(int queue_id, int num_packets, int read_dma_bytes) {

        // construct gpu-command-start message {start, queue ID, number of MMIO packets to read, number of DMA bytes to read}
        uint32_t start_stream   = (uint32_t) 0xFF;
        uint32_t queue          = ((uint32_t) queue_id) & 0xFF;  
        uint32_t size           = ((uint32_t) num_packets) & 0xFF;
        uint32_t dma_bytes      = ((uint32_t) read_dma_bytes) & 0xFF;
        start_stream = (start_stream << 24) | (queue << 16) | (size << 8) | (dma_bytes);

        return start_stream;
    }

    void make_pose_packets(uint32_t* packets, pose_type pose) {
        packets[1] = (uint32_t) pose.position.x();
        packets[2] = (uint32_t) pose.position.y();
        packets[3] = (uint32_t) pose.position.z();
        packets[4] = (uint32_t) pose.orientation.w();
        packets[5] = (uint32_t) pose.orientation.x();
        packets[6] = (uint32_t) pose.orientation.y();
        packets[7] = (uint32_t) pose.orientation.z();
    } 

    long int read_delay_time() {
        // look for one packet containing the amount of time to delay in ns
        while ((reg_read8(GRAPHICS_STATUS) & 0x1) == 0) ;
        return (long int) reg_read32(GRAPHICS_OUT);
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

    static inline uint64_t rdcycle() {
        uint64_t cycles;
        asm volatile ("rdcycle %0" : "=r" (cycles)); // Read cycle counter
        return cycles;
    }

    
    const std::shared_ptr<switchboard>         sb;
    const std::shared_ptr<pose_prediction>     pp;
    const std::shared_ptr<const RelativeClock> _m_clock;

    intptr_t ptr, dma_ptr;
    uint32_t tx_packets[50];
    uint32_t rx_packets[50];

    int        fps{};
    time_point last_fps_update;

    int frame_count = 0;
    record_coalescer mtp_logger;
};
PLUGIN_MAIN(native_renderer)
