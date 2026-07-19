cmake_minimum_required(VERSION 3.20)
project(emender_native_dataplane_transport VERSION 1.0.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CTest)

option(NDP_BUILD_TESTS "Build local libfabric/owner protocol tests" ON)
option(NDP_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBFABRIC REQUIRED IMPORTED_TARGET libfabric>=1.18)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(Threads REQUIRED)

# Standalone transport builds still assemble the same persistent service.  The
# unified native/ build has already created this target.
if(NOT TARGET emender_ndp_service_core)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../src/native_resilient_dataplane
                   ${CMAKE_CURRENT_BINARY_DIR}/local-service)
endif()

add_library(emender_ndp_core STATIC
  src/protocol.cpp
  src/owner.cpp
  src/fabric.cpp)
set_target_properties(emender_ndp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(emender_ndp_core PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
target_include_directories(emender_ndp_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(emender_ndp_core
  PUBLIC PkgConfig::LIBFABRIC OpenSSL::Crypto Threads::Threads)
target_compile_options(emender_ndp_core PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=return-type
  -fno-fast-math -ffp-contract=off)

add_library(emender_ndp_transport SHARED src/transport_c.cpp)
add_library(emender::ndp_transport ALIAS emender_ndp_transport)
target_include_directories(emender_ndp_transport
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(emender_ndp_transport PRIVATE emender_ndp_core)
target_link_options(emender_ndp_transport PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/ndp_transport.map")
target_compile_options(emender_ndp_transport PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=return-type
  -fno-fast-math -ffp-contract=off)
set_target_properties(emender_ndp_transport PROPERTIES
  OUTPUT_NAME emender_ndp_transport
  VERSION 1.0.0
  SOVERSION 1
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

if(NDP_ENABLE_SANITIZERS)
  target_compile_options(emender_ndp_core PRIVATE -fsanitize=address,undefined)
  target_link_options(emender_ndp_core INTERFACE -fsanitize=address,undefined)
  target_compile_options(emender_ndp_transport PRIVATE -fsanitize=address,undefined)
  target_link_options(emender_ndp_transport PUBLIC -fsanitize=address,undefined)
endif()

add_executable(ndp_cxi_service src/service_main.cpp)
target_include_directories(ndp_cxi_service PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ndp_cxi_service PRIVATE
  emender_ndp_transport emender_ndp_core emender_ndp_service_core)
target_compile_options(ndp_cxi_service PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# The Frontier G2 executable is intentionally part of the installed, hashed
# bundle. It uses the same protocol/owner implementation and transport C ABI
# as the persistent service while keeping synthetic dense bytes out of Python.
add_executable(ndp_frontier_2n_gate src/frontier_2n_gate.cpp)
target_include_directories(ndp_frontier_2n_gate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ndp_frontier_2n_gate PRIVATE
  emender_ndp_transport emender_ndp_core)
target_compile_options(ndp_frontier_2n_gate PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=return-type
  -fno-fast-math -ffp-contract=off)

install(TARGETS emender_ndp_transport ndp_cxi_service ndp_frontier_2n_gate
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/emender/ndp_transport.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/emender)

if(BUILD_TESTING AND NDP_BUILD_TESTS)
  add_executable(ndp_protocol_owner_test tests/protocol_owner_test.cpp)
  target_include_directories(ndp_protocol_owner_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(ndp_protocol_owner_test PRIVATE emender_ndp_core)
  target_compile_options(ndp_protocol_owner_test PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
  add_test(NAME ndp_protocol_owner_test COMMAND ndp_protocol_owner_test)

  add_executable(ndp_fabric_multiprocess_test tests/fabric_multiprocess_test.cpp)
  target_include_directories(ndp_fabric_multiprocess_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(ndp_fabric_multiprocess_test PRIVATE emender_ndp_core)
  target_compile_options(ndp_fabric_multiprocess_test PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
  add_test(NAME ndp_fabric_multiprocess_test COMMAND ndp_fabric_multiprocess_test)
  set_tests_properties(ndp_fabric_multiprocess_test PROPERTIES TIMEOUT 45)

  add_executable(ndp_transport_c_abi_test tests/transport_c_abi_test.c)
  set_target_properties(ndp_transport_c_abi_test PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(ndp_transport_c_abi_test PRIVATE emender_ndp_transport)
  target_compile_options(ndp_transport_c_abi_test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME ndp_transport_c_abi_test COMMAND ndp_transport_c_abi_test)
  set_tests_properties(ndp_transport_c_abi_test PROPERTIES TIMEOUT 15)

  add_executable(ndp_service_rpc_integration_test
    tests/service_rpc_integration_test.cpp)
  target_include_directories(ndp_service_rpc_integration_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/native_resilient_dataplane/src)
  # The integration fixture computes exact expected source digests through the
  # service's private optimized SHA wrapper as well as exercising the RPC ABI.
  target_link_libraries(ndp_service_rpc_integration_test PRIVATE
    emender_ndp OpenSSL::Crypto)
  target_compile_options(ndp_service_rpc_integration_test PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=return-type)
  add_test(NAME ndp_service_rpc_integration_test
    COMMAND ndp_service_rpc_integration_test $<TARGET_FILE:ndp_cxi_service>)
  set_tests_properties(ndp_service_rpc_integration_test PROPERTIES TIMEOUT 90)

  add_test(NAME ndp_production_provider_fails_closed
    COMMAND ndp_cxi_service --provider tcp\;ofi_rxm --require-provider cxi --production --probe)
  set_tests_properties(ndp_production_provider_fails_closed PROPERTIES WILL_FAIL TRUE TIMEOUT 15)

  add_test(NAME ndp_test_provider_explicit
    COMMAND ndp_cxi_service --provider tcp\;ofi_rxm --test-only --probe --bind-node 127.0.0.1)
  set_tests_properties(ndp_test_provider_explicit PROPERTIES TIMEOUT 15)

  add_test(NAME ndp_test_provider_explicit_resident_limit
    COMMAND ndp_cxi_service --provider tcp\;ofi_rxm --test-only --probe
      --bind-node 127.0.0.1 --payload-max 4096 --resident-limit 1048576
      --tx-slots 4 --rx-slots 4)
  set_tests_properties(ndp_test_provider_explicit_resident_limit PROPERTIES TIMEOUT 15)

  add_test(NAME ndp_test_mode_cannot_select_cxi
    COMMAND ndp_cxi_service --provider cxi --test-only --probe)
  set_tests_properties(ndp_test_mode_cannot_select_cxi PROPERTIES WILL_FAIL TRUE TIMEOUT 15)

  add_test(NAME ndp_production_env_cannot_weaken
    COMMAND ndp_cxi_service --provider cxi --require-provider cxi --production --probe)
  set_tests_properties(ndp_production_env_cannot_weaken PROPERTIES
    WILL_FAIL TRUE TIMEOUT 15 ENVIRONMENT "FI_PROVIDER=tcp")
endif()
