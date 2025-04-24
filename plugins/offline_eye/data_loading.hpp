#pragma once

#include "illixr/csv_iterator.hpp"
#include "illixr/data_format.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <utility>

typedef unsigned long long ullong;

/*
 * Uncommenting this preprocessor macro makes the offline_cam load each data from the disk as it is needed.
 * Otherwise, we load all of them at the beginning, hold them in memory, and drop them in the queue as needed.
 * Lazy loading has an artificial negative impact on performance which is absent from an online-sensor system.
 * Eager loading deteriorates the startup time and uses more memory.
 */
//#define LAZY

class lazy_load_image {
public:
    lazy_load_image() { }

    lazy_load_image(std::string path)
        : _m_path(std::move(path)) {
#ifndef LAZY
        _m_mat = cv::imread(_m_path, cv::IMREAD_GRAYSCALE);
#endif
    }

    [[nodiscard]] cv::Mat load() const {
#ifdef LAZY
        cv::Mat _m_mat = cv::imread(_m_path, cv::IMREAD_GRAYSCALE);
    #error "Linux scheduler cannot interrupt IO work, so lazy-loading is unadvisable."
#endif
        assert(!_m_mat.empty());
        return _m_mat;
    }

private:
    std::string _m_path;
    cv::Mat     _m_mat;
};

typedef struct {
    float eye_x;
    float eye_y;
} eye_position;

typedef struct {
    lazy_load_image     eye_img;
    eye_position        eye_pos;
} sensor_types;

static std::map<ullong, sensor_types> load_data() {
    const char* illixr_data_c_str = std::getenv("ILLIXR_EYE_DATA");
    if (!illixr_data_c_str) {
        spdlog::get("illixr")->error("[offline_eye] Please define ILLIXR_EYE_DATA");
        ILLIXR::abort();
    }
    std::string illixr_data = std::string{illixr_data_c_str};

    std::map<ullong, sensor_types> data;

    const std::string eye_subpath = "/eye0/data_with_labels.csv";
    std::ifstream     eye_file{illixr_data + eye_subpath};
    if (!eye_file.good()) {
        spdlog::get("illixr")->error("[offline_eye] ${ILLIXR_EYE_DATA} {0} ({1}{0}) is not a good path", eye_subpath, illixr_data);
        ILLIXR::abort();
    }
    for (CSVIterator row{eye_file, 1}; row != CSVIterator{}; ++row) {
        ullong t     = std::stoull(row[0]);
        data[t].eye_img = {illixr_data + "/eye0/data/" + row[1]};

        data[t].eye_pos.eye_x = std::stof(row[2]);
        data[t].eye_pos.eye_y = std::stof(row[3]);
    }


    return data;
}
