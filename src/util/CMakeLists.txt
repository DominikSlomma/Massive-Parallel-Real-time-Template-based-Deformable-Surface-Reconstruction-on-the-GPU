add_library(util_files
            # ${CMAKE_CURRENT_SOURCE_DIR}/config.h
            ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
            # MeshDrawer.cc MeshDrawer.h
            )

# Set output directory of the library
set_target_properties(util_files PROPERTIES
                      OUTPUT_NAME util_files
                      ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
                      LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
# Include directories
target_include_directories(util_files
                           PUBLIC
                           "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/utils/>"
                           "$<INSTALL_INTERFACE:include/>")

# It has to be PUBLIC not PRIVATE!!!!
target_link_libraries(util_files PUBLIC 
                                # pango_display
                                # pango_opengl
                                # ${OpenCV_LIBS}
                                # opencv_imgcodecs
                                # opencv_videoio
                                # opencv_core
                                # opencv_features2d
                                # opencv_calib3d
                                #pangolin
                                # ${CHOLMOD_LIBRARIES} 
                                # ${BLAS_LIBRARIES}
                                # ${LAPACK_LIBRARIES}
                                Eigen3::Eigen
                                Open3D::Open3D
                                yaml-cpp
                                # ${Pangolin_LIBRARIES}
                                # ${PLOTUTILS_LIBRARIES}
                                )

set(UTILS_INCLUDE_INSTALL_DIR ${INCLUDES_DESTINATION}/util_files)

# Install library
install(TARGETS util_files
        EXPORT ${UTILS_TARGETS_EXPORT_NAME}
        RUNTIME DESTINATION ${RUNTIME_DESTINATION}
        LIBRARY DESTINATION ${LIBRARY_DESTINATION}
        ARCHIVE DESTINATION ${ARCHIVE_DESTINATION})

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${UTILS_INCLUDE_INSTALL_DIR})




