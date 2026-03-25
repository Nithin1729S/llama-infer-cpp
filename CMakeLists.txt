cmake_minimum_required(VERSION 3.16)
project(llama_infer)

set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf REQUIRED)
find_package(GTest REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/llama_model.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

add_executable(run_main
    llama_infer_main.cc
    llama_infer.cc
    ${PROTO_SRCS}
)

target_link_libraries(run_main
    ${Protobuf_LIBRARIES}
    pthread
)

add_executable(run_tests
    llama_infer_test.cc
    llama_infer.cc
    ${PROTO_SRCS}
)

target_link_libraries(run_tests
    ${Protobuf_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
    pthread
)