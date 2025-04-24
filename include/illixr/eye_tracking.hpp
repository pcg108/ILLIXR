#pragma once

#include "data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/relative_clock.hpp"

#include <eigen3/Eigen/Geometry>

using namespace ILLIXR;

class eye_tracking_target : public phonebook::service {
public:
    [[nodiscard]] virtual eye_position_type     get_eye_position()                              = 0;
    ~eye_tracking_target() override = default;
};
