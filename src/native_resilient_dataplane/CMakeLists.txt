cmake_minimum_required(VERSION 3.15)
project(emender_native_resilient_dataplane VERSION 1.0.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CTest)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

option(NDP_ENABLE_XPMEM "Enable same-process XPMEM-address ABI registration hook" OFF)
option(NDP_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)

add_library(emender_ndp_service_core STATIC
    src/ndp.cpp
    src/rpc_server.cpp)
set_target_properties(emender_ndp_service_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
target_compile_features(emender_ndp_service_core PUBLIC cxx_std_17)
target_include_directories(emender_ndp_service_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(emender_ndp_service_core PUBLIC Threads::Threads OpenSSL::Crypto)
target_compile_options(emender_ndp_service_core PRIVATE
    -Wall -Wextra -Wpedantic
    -fno-fast-math -ffp-contract=off -frounding-math)

add_library(emender_ndp SHARED src/client.cpp)
add_library(Emender::ndp ALIAS emender_ndp)
target_compile_features(emender_ndp PUBLIC cxx_std_17)
set_target_properties(emender_ndp PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME emender_ndp
    VERSION 1.0.0
    SOVERSION 1
)
target_include_directories(emender_ndp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(emender_ndp PRIVATE Threads::Threads OpenSSL::Crypto)
target_compile_options(emender_ndp PRIVATE
    -Wall -Wextra -Wpedantic
    -fno-fast-math -ffp-contract=off -frounding-math
)
if(NDP_ENABLE_SANITIZERS)
    target_compile_options(emender_ndp_service_core PRIVATE -fsanitize=address,undefined)
    target_link_options(emender_ndp_service_core INTERFACE -fsanitize=address,undefined)
    target_compile_options(emender_ndp PRIVATE -fsanitize=address,undefined)
    target_link_options(emender_ndp PUBLIC -fsanitize=address,undefined)
endif()
if(NDP_ENABLE_XPMEM)
    target_compile_definitions(emender_ndp_service_core PRIVATE NDP_ENABLE_XPMEM=1)
else()
    target_compile_definitions(emender_ndp_service_core PRIVATE NDP_ENABLE_XPMEM=0)
endif()

install(TARGETS emender_ndp
    EXPORT EmenderNdpTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES include/emender/ndp.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/emender)

if(BUILD_TESTING)
    add_executable(ndp_local_abi_test tests/abi_test.c)
    target_link_libraries(ndp_local_abi_test PRIVATE emender_ndp)
    # The C ABI test still needs the C++ sanitizer runtime when the shared
    # implementation is instrumented.
    set_target_properties(ndp_local_abi_test PROPERTIES
        C_STANDARD 11 C_EXTENSIONS OFF LINKER_LANGUAGE CXX)
    add_test(NAME ndp_local_abi_test COMMAND ndp_local_abi_test)
endif()
