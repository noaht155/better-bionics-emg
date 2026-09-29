# PlatformIO's build_src_filter is ignored under ESP-IDF, so only src/espidf is compiled here
FILE(GLOB_RECURSE app_sources ${CMAKE_SOURCE_DIR}/src/espidf/*.*)

idf_component_register(SRCS ${app_sources})
