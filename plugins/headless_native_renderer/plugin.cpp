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
#include "illixr/gtsam_integrator.hpp"
#include "illixr/read_hpm.h"

#include "illixr/plugin.hpp"

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

class native_renderer : public plugin {
public:
    native_renderer(std::string name_, phonebook* pb)
        : plugin{std::move(name_), pb}
        , sb{pb->lookup_impl<switchboard>()}
        , pp{pb->lookup_impl<pose_prediction>()}
        , gpu{pb->lookup_impl<gpu_model>()}
        , et{pb->lookup_impl<eye_tracking_target>()}
        , gint{pb->lookup_impl<gtsam_integrator>()}
        , _m_clock{pb->lookup_impl<RelativeClock>()}
        , last_fps_update{std::chrono::duration<long, std::nano>{0}}
        , mtp_logger{record_logger_} {
        spdlogger(std::getenv("NATIVE_RENDERER_LOG_LEVEL"));

        sb->schedule<imu_type>(id, "imu", [&](const switchboard::ptr<const imu_type>& datum, size_t) {
            callback(datum);
        });

    }


    /**
     * @brief Executes one iteration of the plugin's main loop.
     */
    void callback(const switchboard::ptr<const imu_type>& datum) {

        gint->callback(datum);

        read_counters(counters_before);

        imu_sample_count += 1;
        if (imu_sample_count % 10 != 0) {
            return;
        }

        uint32_t response_buffer[5];

        eye_position_type send_eye_pos  = et->get_eye_position(); // eye_position_type{_m_clock->now(), 0.0, 0.0}; 

        uint64_t before_render_pose = rdcycle();

            auto render_pose = pp->get_fast_pose();

            // if (render_pose.pose.position.x() != 0) {
                // send_eye_pos = et->get_eye_position();
                // std::cout << "pose: " << render_pose.pose.position.x() << " " << render_pose.pose.position.y() << " " << render_pose.pose.position.z() << std::endl;
            // }
            

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
        // std::cout << "MTP: " << after_tw - before_render_pose << std::endl;

        mtp_logger.log(record{mtp_record,
            {
                {(size_t) (after_render_pose - before_render_pose)},
                {(size_t) (after_render - before_render)},
                {(size_t) (after_tw_pose - before_tw_pose)},
                {(size_t) (after_tw - before_tw)},
                {_m_clock->now() - timewarp_pose.pose.sensor_time},
            }});

        read_counters(counters_after);
        std::cout << "headless_native_renderer: " << diff_to_string(counters_after, counters_before) << std::endl;

    }

private:

    double counters_before[29] = {0.0};
    double counters_after[29] = {0.0};

    static inline uint64_t rdcycle() {
        uint64_t cycles;
        asm volatile ("rdcycle %0" : "=r" (cycles)); // Read cycle counter
        return cycles;
    }

    
    const std::shared_ptr<switchboard>          sb;
    const std::shared_ptr<pose_prediction>      pp;
    const std::shared_ptr<gpu_model>            gpu;
    const std::shared_ptr<eye_tracking_target>  et;
    const std::shared_ptr<gtsam_integrator>     gint;
    const std::shared_ptr<const RelativeClock> _m_clock;


    int        fps{};
    time_point last_fps_update;

    int frame_count = 0;
    record_coalescer mtp_logger;

    int imu_sample_count = 0;
};
PLUGIN_MAIN(native_renderer)
