# module to build and install the eye tracking ILLIXR plugin

set(PLUGIN_NAME plugin.eye_tracking_target${ILLIXR_BUILD_SUFFIX})

find_package(onnxruntime REQUIRED)

# source files, listed individually so that any changes will trigger a rebuild
add_library(${PLUGIN_NAME} SHARED plugin.cpp
        ../../include/illixr/plugin.hpp
        ../../include/illixr/opencv_data_types.hpp
        ../../include/illixr/data_format.hpp
        ../../include/illixr/phonebook.hpp
        ../../include/illixr/eye_tracking.hpp
        )

if(BUILD_OPENCV)
        add_dependencies(${PLUGIN_NAME} OpenCV_Viz)
endif()

target_include_directories(${PLUGIN_NAME} PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR} 
        ${OpenCV_INCLUDE_DIRS} 
        ${Protobuf_INCLUDE_DIRS} 
        ${ILLIXR_SOURCE_DIR}/include
        ${onnxruntime_INCLUDE_DIRS})

target_include_directories(${PLUGIN_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ILLIXR_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS})
target_link_libraries(${PLUGIN_NAME} 
        ${OpenCV_LIBRARIES}
        ${Eigen3_LIBRARIES} 
        spdlog::spdlog
        ${onnxruntime_LIBRARIES})

target_compile_features(${PLUGIN_NAME} PRIVATE cxx_std_17)

install(TARGETS ${PLUGIN_NAME} DESTINATION lib)

