set(PLUGIN_NAME plugin.eye_tracking_target${ILLIXR_BUILD_SUFFIX})

# Create interface library for C headers
add_library(ritnet_c_headers INTERFACE)
target_include_directories(ritnet_c_headers INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compile C code as static library
add_library(ritnet_c STATIC
    include/ritnet.h
    ritnet.c
)
target_include_directories(ritnet_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_source_files_properties(ritnet.c PROPERTIES LANGUAGE C)

# Main plugin target
add_library(${PLUGIN_NAME} SHARED 
    plugin.cpp
    ../../include/illixr/plugin.hpp
    ../../include/illixr/opencv_data_types.hpp
    ../../include/illixr/data_format.hpp
    ../../include/illixr/phonebook.hpp
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
    ritnet_c
    ritnet_c_headers
    ${OpenCV_LIBRARIES}
    ${Eigen3_LIBRARIES}
    ${onnxruntime_LIBRARIES}
    spdlog::spdlog
)

# Handle C/C++ compatibility
target_compile_options(${PLUGIN_NAME} PRIVATE 
    -Wno-c11-extensions          # For C11 features in C++
    -Wno-vla-extension           # For VLA in headers
)

target_compile_features(${PLUGIN_NAME} PRIVATE cxx_std_17)

install(TARGETS ${PLUGIN_NAME} DESTINATION lib)