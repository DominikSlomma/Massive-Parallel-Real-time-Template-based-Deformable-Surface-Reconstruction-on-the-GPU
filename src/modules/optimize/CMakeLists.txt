# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
        #        ${CMAKE_CURRENT_SOURCE_DIR}/Optimizer.cc ${CMAKE_CURRENT_SOURCE_DIR}/Optimizer.h
               ${CMAKE_CURRENT_SOURCE_DIR}/OptimizerWithoutMiddlePoint.cc ${CMAKE_CURRENT_SOURCE_DIR}/OptimizerWithoutMiddlePoint.h
               ${CMAKE_CURRENT_SOURCE_DIR}/OptimizerDistanceOnly.cc ${CMAKE_CURRENT_SOURCE_DIR}/OptimizerDistanceOnly.h
               ${CMAKE_CURRENT_SOURCE_DIR}/optGPU.cu ${CMAKE_CURRENT_SOURCE_DIR}/optGPU.cuh
               ${CMAKE_CURRENT_SOURCE_DIR}/gauss_newton.cu ${CMAKE_CURRENT_SOURCE_DIR}/gauss_newton.cuh
        #        ${CMAKE_CURRENT_SOURCE_DIR}/OptimizerVB.cc ${CMAKE_CURRENT_SOURCE_DIR}/OptimizerVB.h
               )

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${STBR_INCLUDE_INSTALL_DIR}/optimize)