#include "data_loading.hpp"
#include "illixr/opencv_data_types.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/relative_clock.hpp"
#include "illixr/threadloop.hpp"

#include "illixr/offline_cam.hpp"

#include <chrono>
#include <shared_mutex>
#include <thread>

using namespace ILLIXR;

class offline_cam_impl : public offline_cam {
public:
    explicit offline_cam_impl(const phonebook* const pb)
    : sb{pb->lookup_impl<switchboard>()}
    , _m_sensor_data{load_data()}
    , dataset_first_time{_m_sensor_data.cbegin()->first}
    , last_ts{0}
    , _m_rtc{pb->lookup_impl<RelativeClock>()}
    , next_row{_m_sensor_data.cbegin()} 
    { }
        
    std::optional<cam_type> get_cam_reading(time_point imu_time) {
        ullong lookup_time = imu_time.time_since_epoch().count();

        if (lookup_time < dataset_first_time) {
            return std::nullopt;
        }

        std::map<ullong, sensor_types>::const_iterator nearest_row;
        auto after_nearest_row = _m_sensor_data.find(lookup_time);
        if (after_nearest_row == _m_sensor_data.cend()) {
            return std::nullopt;
        }

        if (after_nearest_row == _m_sensor_data.cend()) {
            // Handling the last camera images. There's no more rows after the nearest_row, so we set after_nearest_row
            // to be nearest_row to avoiding sleeping at the end.
            nearest_row       = std::prev(after_nearest_row, 1);
            after_nearest_row = nearest_row;
            // We are running out of the dataset and the loop will stop next time.
            internal_stop();
        } else if (after_nearest_row == _m_sensor_data.cbegin()) {
            // Should not happen because lookup_time is bigger than dataset_first_time
        } else {
            // Most recent
            nearest_row = std::prev(after_nearest_row, 1);
        }

        std::cout << " offline_cam: " << nearest_row->first << " from_imu: " << lookup_time << std::endl;

        if (last_ts != nearest_row->first) {
            last_ts = nearest_row->first;

            auto img0 = nearest_row->second.cam0.load();
            auto img1 = nearest_row->second.cam1.load();

            time_point expected_real_time_given_dataset_time(
                std::chrono::duration<long, std::nano>{nearest_row->first - dataset_first_time});

            return cam_type{
                expected_real_time_given_dataset_time,
                img0,
                img1,
            }
        }
    }



private:
    const std::shared_ptr<switchboard>             sb;
    const std::map<ullong, sensor_types>           _m_sensor_data;
    ullong                                         dataset_first_time;
    ullong                                         last_ts;
    std::shared_ptr<RelativeClock>                 _m_rtc;
    std::map<ullong, sensor_types>::const_iterator next_row;

};

class offline_cam_plugin : public plugin {
public:
    offline_cam_plugin(const std::string& name, phonebook* pb)
        : plugin{name, pb} {
        pb->register_impl<offline_cam>(
            std::static_pointer_cast<offline_cam>(std::make_shared<offline_cam_impl>(pb)));
    }
};

PLUGIN_MAIN(offline_cam_plugin);


