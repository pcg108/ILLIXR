#pragma once

#include "data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/relative_clock.hpp"

#include <eigen3/Eigen/Geometry>

using namespace ILLIXR;

class gtsam_integrator : public phonebook::service {
public:
    [[nodiscard]] virtual void     callback() const                             = 0;

    ~gtsam_integrator() override = default;
};
