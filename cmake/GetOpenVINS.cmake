set(OPENVINS_CMAKE_ARGS "")

# if building on CentOS make sure we use the correct OpenCV
if(HAVE_CENTOS)
    set(OPENVINS_CMAKE_ARGS "-DOpenCV_DIR=${OpenCV_DIR}")
endif()

EXTERNALPROJECT_ADD(OpenVINS
        GIT_REPOSITORY https://github.com/pcg108/open_vins.git   # Git repo for source code
        GIT_TAG 1c730a466b74e3243b179d8a50c0f8de42137c27  
	PREFIX ${CMAKE_BINARY_DIR}/_deps/OpenVINS                # the build directory
        DEPENDS ${OpenCV_DEP_STR}   # dependencies of this module
        #arguments to pass to CMake
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DCMAKE_CXX_FLAGS=-L${CMAKE_INSTALL_PREFIX}/lib -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DILLIXR_ROOT=${PROJECT_SOURCE_DIR}/include -DILLIXR_BUILD_SUFFIX=${ILLIXR_BUILD_SUFFIX} -DILLIXR_INTEGRATION=ON ${OPENVINS_CMAKE_ARGS}
        )
