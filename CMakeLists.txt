cmake_minimum_required(VERSION 3.28)


find_program(UNBUFFER_EXECUTABLE unbuffer)
if(UNBUFFER_EXECUTABLE)
    message(STATUS "Using unbuffer for CUDA compiler output color.")
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${UNBUFFER_EXECUTABLE})
else()
    message(WARNING "unbuffer not found. CUDA compiler output might not be colored.")
endif()


project(course_project LANGUAGES C CXX)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    enable_language(CUDA)
endif()


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(IS_VISUAL_STUDIO "false")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5")
else()
    set(CUDA_DIR /opt/cuda/)
endif()
include_directories(${CUDA_DIR}/include/)

add_executable(full)


if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(full
            PRIVATE main_test.c
            PRIVATE reduction_test.c
            PRIVATE scan_test.c
            PRIVATE reduction_avx.c
            PRIVATE reduction.cu
            PRIVATE scan.cu
    )
endif()


message(STATUS "Linker: ${CMAKE_LINKER}")


set(CMAKE_COLOR_DIAGNOSTICS ON)


set_target_properties(full PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75")


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(full PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info -O3 -Wall -Xcompiler -fdiagnostics-color=always>)
    target_compile_options(full PRIVATE $<$<COMPILE_LANGUAGE:C>:-mavx2 -O3 -Wall>)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(IS_VISUAL_STUDIO STREQUAL "true")
        target_compile_options(full PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-G -g -O0>)
    else()
        target_compile_options(full PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info -g -G>)
        target_compile_options(full PRIVATE $<$<COMPILE_LANGUAGE:C>:-mavx2>)
    endif()
endif()

target_compile_definitions(full PRIVATE BENCHMARK)


set(COMPILE_COMMANDS_EXPORT_DIR "${CMAKE_SOURCE_DIR}")

add_custom_command(
    TARGET full POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${COMPILE_COMMANDS_EXPORT_DIR}/compile_commands.json
)


add_custom_command(
    TARGET full POST_BUILD
    COMMAND sh -c "echo '\n'; printf '\\e[32m'; tail -n +2 '${CMAKE_BINARY_DIR}/.ninja_log' | cat -n; printf '\\e[0m'"
    VERBATIM
)


add_custom_command(
    TARGET full POST_BUILD
    COMMAND sh -c "echo '\n'; ninja -t commands full | sed 'G' | nl"
    VERBATIM
)


add_library(component_lib STATIC
  scan.cu
)


add_library(multivex SHARED
    scan.cu
)


add_subdirectory(tests)

