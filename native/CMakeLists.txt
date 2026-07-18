cmake_minimum_required(VERSION 3.20)
project(emender_native_resilient_dataplane_bundle VERSION 1.0.0 LANGUAGES C CXX)

# One authoritative configure/install surface keeps the local and network ABI
# artifacts on the same source commit and compiler configuration.  The child
# projects retain their focused standalone builds for component development.
include(CTest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/native_resilient_dataplane
                 ${CMAKE_CURRENT_BINARY_DIR}/local)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dataplane
                 ${CMAKE_CURRENT_BINARY_DIR}/transport)
