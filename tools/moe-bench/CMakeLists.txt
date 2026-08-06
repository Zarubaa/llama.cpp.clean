set(TARGET llama-moe-bench)

add_executable(${TARGET} main.cpp page-cache.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(SERVER_TARGET llama-moe-bench-server)

    add_executable(${SERVER_TARGET} server.cpp)
    target_link_libraries(${SERVER_TARGET} PRIVATE cpp-httplib Threads::Threads)
    target_include_directories(${SERVER_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor)
    target_compile_features(${SERVER_TARGET} PRIVATE cxx_std_17)
endif()

if (LLAMA_MOE_OFFLOAD)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_MOE_OFFLOAD)
endif()

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE psapi)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        install(TARGETS ${SERVER_TARGET} RUNTIME)
    endif()
endif()
