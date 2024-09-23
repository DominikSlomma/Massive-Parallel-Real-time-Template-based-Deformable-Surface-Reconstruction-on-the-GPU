# add_executable(cuda_opt_test main.cu main.cuh)
# add_executable(cuda_opt_test2 optim.cu optim.cuh)

# set_target_properties(cuda_opt_test PROPERTIES
#     CUDA_STANDARD 11
#     CUDA_STANDARD_REQUIRED ON
# )
# set_target_properties(cuda_opt_test2 PROPERTIES
#     CUDA_STANDARD 11
#     CUDA_STANDARD_REQUIRED ON
# )

# # Enable fast math optimizations for CUDA
# target_compile_options(cuda_opt_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
# target_compile_options(cuda_opt_test2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)

# # Link Eigen3 to the target
# target_link_libraries(cuda_opt_test Eigen3::Eigen ${SUITESPARSE_LIBRARIES}
# ${CHOLMOD_LIBRARIES} 
# ${BLAS_LIBRARIES}
# ${LAPACK_LIBRARIES})          

# # Link Eigen3 to the target
# target_link_libraries(cuda_opt_test2 Eigen3::Eigen ${SUITESPARSE_LIBRARIES} 
# Open3D::Open3D
# ${CHOLMOD_LIBRARIES} 
# ${BLAS_LIBRARIES}
# ${LAPACK_LIBRARIES})                                                     

# CUDA Architektur setzen
set(CUDA_ARCHITECTURES 70 75 80)

add_executable(cuda_opt_test main.cu main.cuh)
add_executable(cuda_opt_test2 optim.cu optim.cuh)
add_executable(cuda_opt_test3 optim_kernelStreams.cu optim_kernelStreams.cuh)

set_target_properties(cuda_opt_test PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
)
set_target_properties(cuda_opt_test2 PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
)
set_target_properties(cuda_opt_test3 PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
)

# Fast Math Optimierungen für CUDA aktivieren
target_compile_options(cuda_opt_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
target_compile_options(cuda_opt_test2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
target_compile_options(cuda_opt_test3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)

# Include Directories hinzufügen
target_include_directories(cuda_opt_test PRIVATE ${EIGEN3_INCLUDE_DIR} ${SUITESPARSE_INCLUDE_DIR} ${OPEN3D_INCLUDE_DIR})
target_include_directories(cuda_opt_test2 PRIVATE ${EIGEN3_INCLUDE_DIR} ${SUITESPARSE_INCLUDE_DIR} ${OPEN3D_INCLUDE_DIR})
target_include_directories(cuda_opt_test3 PRIVATE ${EIGEN3_INCLUDE_DIR} ${SUITESPARSE_INCLUDE_DIR} ${OPEN3D_INCLUDE_DIR})

# Libraries verlinken
target_link_libraries(cuda_opt_test Eigen3::Eigen ${SUITESPARSE_LIBRARIES} ${CHOLMOD_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(cuda_opt_test2 Eigen3::Eigen ${SUITESPARSE_LIBRARIES} ${CHOLMOD_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${CUDA_LIBRARIES} cusolver cudart cusparse)
target_link_libraries(cuda_opt_test3 Eigen3::Eigen ${SUITESPARSE_LIBRARIES} ${CHOLMOD_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${CUDA_LIBRARIES} cusolver cudart cusparse ${SUITESPARSE_LIBRARIES}
${CHOLMOD_LIBRARIES} 
${BLAS_LIBRARIES}
${LAPACK_LIBRARIES}
Eigen3::Eigen )
 