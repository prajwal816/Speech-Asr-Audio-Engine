cmake_minimum_required(VERSION 3.14)
project(AudioFeatures LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Library ──────────────────────────────────────────────────
add_library(audio_features STATIC
    audio_features.cpp
)
target_include_directories(audio_features PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ─── Demo executable ─────────────────────────────────────────
add_executable(audio_demo main.cpp)
target_link_libraries(audio_demo PRIVATE audio_features)

# ─── Compiler warnings ───────────────────────────────────────
if(MSVC)
    target_compile_options(audio_features PRIVATE /W4)
    target_compile_options(audio_demo PRIVATE /W4)
else()
    target_compile_options(audio_features PRIVATE -Wall -Wextra -O2)
    target_compile_options(audio_demo PRIVATE -Wall -Wextra -O2)
endif()
