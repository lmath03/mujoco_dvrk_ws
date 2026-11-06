cmake_minimum_required(VERSION 3.0.2)
project(my_pcl_test)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
)

find_package(PCL REQUIRED)

# Questo DEVE stare prima di add_executable
catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Qui creiamo l'eseguibile
add_executable(example src/example.cpp)

target_link_libraries(example
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)