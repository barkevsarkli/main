cmake_minimum_required(VERSION 3.20)
project(version6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- floating-point policy -------------------------------------------------------------
# No fast-math, no FMA contraction, no OpenMP: every run must be bit-reproducible and the
# summation order is fixed in the source. The default per-config flags are replaced so the
# recorded flags are exactly the flags used.
include(CheckCXXCompilerFlag)
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g")
if(MSVC)
  # MSVC and clang-cl. /arch:AVX2 only exists for x64 targets.
  set(V6_FLAGS /O2 /fp:precise /EHsc)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "ARM|arm|aarch64")
    list(APPEND V6_FLAGS /arch:AVX2)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND V6_FLAGS -ffp-contract=off)
  endif()
else()
  # -march=native is rejected by GCC on aarch64 (Graviton); -mcpu=native is its equivalent.
  check_cxx_compiler_flag(-march=native V6_HAS_MARCH_NATIVE)
  check_cxx_compiler_flag(-mcpu=native V6_HAS_MCPU_NATIVE)
  set(V6_FLAGS -O3)
  if(V6_HAS_MARCH_NATIVE)
    list(APPEND V6_FLAGS -march=native)
  elseif(V6_HAS_MCPU_NATIVE)
    list(APPEND V6_FLAGS -mcpu=native)
  endif()
  list(APPEND V6_FLAGS -ffp-contract=off -fno-fast-math)
endif()
if(V6_EXTRA_FLAGS)
  list(APPEND V6_FLAGS ${V6_EXTRA_FLAGS})
endif()
add_compile_options(${V6_FLAGS})
string(REPLACE ";" " " V6_FLAGS_STR "${V6_FLAGS}")

# ---- provenance header (regenerated on every build, rewritten only when it changes) ----
set(V6_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
add_custom_target(v6_provenance ALL
  COMMAND ${CMAKE_COMMAND}
          -DSRC_DIR=${CMAKE_SOURCE_DIR}
          -DOUT=${V6_GEN_DIR}/provenance.h
          "-DCOMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
          "-DFLAGS=${V6_FLAGS_STR}"
          "-DPLATFORM=${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
          -P ${CMAKE_SOURCE_DIR}/cmake/gen_provenance.cmake
  BYPRODUCTS ${V6_GEN_DIR}/provenance.h
  COMMENT "provenance")

# ---- targets ---------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${cfg} CFG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_SOURCE_DIR}/bin)
endforeach()

add_library(v6core STATIC
  src/cifar10_data.cpp src/dataset.cpp src/kernels.cpp src/mapped_file.cpp
  src/mnist_data.cpp src/net.cpp src/stats.cpp src/util.cpp)
target_include_directories(v6core PUBLIC src ${V6_GEN_DIR})
add_dependencies(v6core v6_provenance)

add_executable(main src/main.cpp)
target_link_libraries(main v6core)

add_executable(kernel_dump tools/kernel_dump.cpp)
target_link_libraries(kernel_dump v6core)

enable_testing()
foreach(t test_rng test_loader test_conv_reference test_gradcheck test_kernels test_misc)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} v6core)
  target_include_directories(${t} PRIVATE tests)
endforeach()
add_test(NAME rng            COMMAND test_rng)
add_test(NAME conv_reference COMMAND test_conv_reference)
add_test(NAME gradcheck      COMMAND test_gradcheck)
add_test(NAME misc           COMMAND test_misc)
add_test(NAME loader         COMMAND test_loader ${CMAKE_SOURCE_DIR}/..)
add_test(NAME kernels        COMMAND test_kernels ${CMAKE_SOURCE_DIR}/kernels)
