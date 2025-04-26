#pragma once

#include "data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/relative_clock.hpp"

using namespace ILLIXR;

class gpu_model : public phonebook::service {
public:
    [[nodiscard]] virtual void     send_gpu_render_message(fast_pose_type current_pose, eye_position_type eye_pos, int queue_id, int read_dma_bytes)  = 0;
    [[nodiscard]] virtual void     send_gpu_compute_message(int queue_id, int read_dma_bytes)                                                         = 0;
    [[nodiscard]] virtual void     copy_to_dma(void* data, int bytes)                                                                                 = 0;

    ~gpu_model() override = default;
};
