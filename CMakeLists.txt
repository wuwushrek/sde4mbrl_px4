cmake_minimum_required(VERSION 3.0.2)
project(mpc4px4)


find_package(catkin_simple REQUIRED)

# Seup python package
catkin_python_setup()

# Build whatever need to be built
catkin_simple(ALL_DEPS_REQUIRED)

set(CMAKE_MACOSX_RPATH 0)
add_definitions(-std=c++11)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

cs_add_executable(traj_gen_client
  mpc4px4/trajectory_generation/traj_gen_client.cpp
)

cs_add_executable(geometric_controller
  mpc4px4/geometric_controller/geometric_controller.cpp
)
target_link_libraries(geometric_controller yaml-cpp)

##########
# EXPORT #
##########
cs_install()
cs_export()