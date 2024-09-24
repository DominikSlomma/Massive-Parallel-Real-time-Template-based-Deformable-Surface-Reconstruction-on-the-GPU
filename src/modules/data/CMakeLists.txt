# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/Extractor.cc ${CMAKE_CURRENT_SOURCE_DIR}/Extractor.h
               ${CMAKE_CURRENT_SOURCE_DIR}/database.cc ${CMAKE_CURRENT_SOURCE_DIR}/database.h
               )

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${STBR_INCLUDE_INSTALL_DIR}/data)