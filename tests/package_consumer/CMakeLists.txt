cmake_minimum_required(VERSION 3.24)
project(neurelease_package_consumer LANGUAGES C)

find_package(neurelease CONFIG REQUIRED)
add_executable(package_consumer main.c)
target_link_libraries(package_consumer PRIVATE neurelease::c)
