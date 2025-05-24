set(PLUGIN_NAME plugin.eye_tracking_target${ILLIXR_BUILD_SUFFIX})


find_package(onnxruntime REQUIRED)


# Main plugin target
add_library(${PLUGIN_NAME} SHARED 
    plugin.cpp
    ../../include/illixr/plugin.hpp
    ../../include/illixr/opencv_data_types.hpp
    ../../include/illixr/data_format.hpp
    ../../include/illixr/phonebook.hpp
)

add_library(gemmini STATIC IMPORTED)
set_target_properties(gemmini PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/gemmini/libgemmini.a
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/gemmini/include
)

if(BUILD_OPENCV)
    add_dependencies(${PLUGIN_NAME} OpenCV_Viz)
endif()

# Link C library and handle C/C++ interop
target_include_directories(${PLUGIN_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${onnxruntime_INCLUDE_DIRS}
    ${ILLIXR_SOURCE_DIR}/include
)

target_include_directories(${PLUGIN_NAME} PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${ILLIXR_SOURCE_DIR}/include
    ${Eigen3_INCLUDE_DIRS}
)


target_link_libraries(${PLUGIN_NAME}
    ${OpenCV_LIBRARIES}
    ${Eigen3_LIBRARIES}
    ${onnxruntime_LIBRARIES}
    spdlog::spdlog
    gemmini
)

target_compile_features(${PLUGIN_NAME} PRIVATE cxx_std_17)

install(TARGETS ${PLUGIN_NAME} DESTINATION lib)