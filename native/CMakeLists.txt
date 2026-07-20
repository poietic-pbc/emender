cmake_minimum_required(VERSION 3.20)
project(emender_native_resilient_dataplane_bundle VERSION 1.0.0 LANGUAGES C CXX)

# One authoritative configure/install surface keeps the local and network ABI
# artifacts on the same source commit and compiler configuration.  The child
# projects retain their focused standalone builds for component development.
include(CTest)
include(GNUInstallDirs)

# CMake normally embeds an absolute build RPATH and edits it in-place during
# installation.  The vacated string-table padding then reflects the absolute
# build-directory length even though the effective installed RPATH is the
# same.  Use an origin-relative build RPATH and one fixed installed RPATH so
# differently shaped clean build directories produce identical package bytes.
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(NDP_ROCM_RUNTIME_DIR "" CACHE PATH
  "Reviewed ROCm runtime library directory embedded in Frontier binaries")
set(NDP_FABRIC_RUNTIME_DIR "" CACHE PATH
  "Reviewed Frontier libfabric runtime directory embedded in Frontier binaries")
set(_emender_native_install_rpath "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
if(NDP_ROCM_RUNTIME_DIR)
  get_filename_component(NDP_ROCM_RUNTIME_DIR "${NDP_ROCM_RUNTIME_DIR}" REALPATH)
  if(NOT IS_ABSOLUTE "${NDP_ROCM_RUNTIME_DIR}" OR
     NOT EXISTS "${NDP_ROCM_RUNTIME_DIR}/libamdhip64.so.7")
    message(FATAL_ERROR
      "NDP_ROCM_RUNTIME_DIR must be an absolute directory containing libamdhip64.so.7")
  endif()
  list(APPEND _emender_native_install_rpath "${NDP_ROCM_RUNTIME_DIR}")
endif()
if(NDP_FABRIC_RUNTIME_DIR)
  get_filename_component(NDP_FABRIC_RUNTIME_DIR "${NDP_FABRIC_RUNTIME_DIR}" REALPATH)
  if(NOT IS_ABSOLUTE "${NDP_FABRIC_RUNTIME_DIR}" OR
     NOT EXISTS "${NDP_FABRIC_RUNTIME_DIR}/libfabric.so.1")
    message(FATAL_ERROR
      "NDP_FABRIC_RUNTIME_DIR must be an absolute directory containing libfabric.so.1")
  endif()
  list(APPEND _emender_native_install_rpath "${NDP_FABRIC_RUNTIME_DIR}")
endif()
set(CMAKE_INSTALL_RPATH "${_emender_native_install_rpath}")

# RelWithDebInfo is the retained Frontier package configuration.  Without
# prefix maps, DWARF records CMAKE_BINARY_DIR and otherwise-identical clean
# builds have different bytes/build IDs solely because their build directories
# differ.  Use stable virtual roots for the unified bundle while retaining
# useful source-relative debug paths.  The two child projects may still be
# configured alone for component development; the canonical package is this
# unified surface.
get_filename_component(EMENDER_NATIVE_SOURCE_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/.." REALPATH)
add_compile_options(
  "-ffile-prefix-map=${EMENDER_NATIVE_SOURCE_ROOT}=/usr/src/emender"
  "-fdebug-prefix-map=${EMENDER_NATIVE_SOURCE_ROOT}=/usr/src/emender"
  "-fmacro-prefix-map=${EMENDER_NATIVE_SOURCE_ROOT}=/usr/src/emender"
  "-ffile-prefix-map=${CMAKE_BINARY_DIR}=/usr/src/emender-build"
  "-fdebug-prefix-map=${CMAKE_BINARY_DIR}=/usr/src/emender-build"
  "-fmacro-prefix-map=${CMAKE_BINARY_DIR}=/usr/src/emender-build")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/native_resilient_dataplane
                 ${CMAKE_CURRENT_BINARY_DIR}/local)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dataplane
                 ${CMAKE_CURRENT_BINARY_DIR}/transport)
