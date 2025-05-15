#pragma once

#include "data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/relative_clock.hpp"

using namespace ILLIXR;

typedef unsigned long long ullong;

class offline_cam : public phonebook::service {
public:
    [[nodiscard]] virtual cam_type  get_cam_reading(ullong imu_time)  = 0;

    ~offline_cam() override = default;
};
