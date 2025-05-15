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
#include "illixr/gpu_model.hpp"
#include "illixr/eye_tracking_target.hpp"
#include "illixr/switchboard.hpp"
#include "illixr/threadloop.hpp"
#include "illixr/vk_util/headless_sink.hpp"
#include "illixr/vk_util/render_pass.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "illixr/gl_util/lib/tiny_obj_loader.h"



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
        , gpu{pb->lookup_impl<gpu_model>()}
        , et{pb->lookup_impl<eye_tracking_target>()}
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

    }

    /**
     * @brief Executes one iteration of the plugin's main loop.
     */
    void _p_one_iteration() override {

        uint32_t response_buffer[5];

        eye_position_type send_eye_pos  = eye_position_type{_m_clock->now(), 0.0, 0.0}; // et->get_eye_position();

        uint64_t before_render_pose = rdcycle();

            auto render_pose = pp->get_fast_pose();
            std::cout << "pose: " << render_pose.pose.position.x() << " " << render_pose.pose.position.y() << " " << render_pose.pose.position.z() << std::endl;

        uint64_t after_render_pose = rdcycle();

        uint64_t before_render = rdcycle();

            gpu->send_gpu_render_message(render_pose, send_eye_pos, 0, 0, 1, response_buffer);

        uint64_t after_render = rdcycle();

        uint64_t before_tw_pose = rdcycle();

            auto timewarp_pose = pp->get_fast_pose();

        uint64_t after_tw_pose = rdcycle();

        uint64_t before_tw = rdcycle();

            gpu->send_gpu_render_message(timewarp_pose, send_eye_pos, 1, 0, 2, response_buffer);

        uint64_t after_tw = rdcycle();

        // std::cout << "MTP: " << duration2double<std::milli>(_m_clock->now() - timewarp_pose.pose.sensor_time) << std::endl;
        std::cout << "MTP: " << after_tw - before_render_pose << std::endl;

        mtp_logger.log(record{mtp_record,
            {
                {(size_t) (after_render_pose - before_render_pose)},
                {(size_t) (after_render - before_render)},
                {(size_t) (after_tw_pose - before_tw_pose)},
                {(size_t) (after_tw - before_tw)},
                {_m_clock->now() - timewarp_pose.pose.sensor_time},
            }});

    }

private:

    

    static inline uint64_t rdcycle() {
        uint64_t cycles;
        asm volatile ("rdcycle %0" : "=r" (cycles)); // Read cycle counter
        return cycles;
    }

    
    const std::shared_ptr<switchboard>          sb;
    const std::shared_ptr<pose_prediction>      pp;
    const std::shared_ptr<gpu_model>            gpu;
    const std::shared_ptr<eye_tracking_target>  et;
    const std::shared_ptr<const RelativeClock> _m_clock;


    int        fps{};
    time_point last_fps_update;

    int frame_count = 0;
    record_coalescer mtp_logger;
};
PLUGIN_MAIN(native_renderer)
