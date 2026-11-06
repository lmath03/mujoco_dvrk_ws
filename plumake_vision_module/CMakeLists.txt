cmake_minimum_required(VERSION 3.0.2)
project(plumake_vision_module)

## Compilazione moderna C++17
add_compile_options(-std=c++17)

## 1. Trova i pacchetti ROS (Catkin)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  std_srvs
  geometry_msgs
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  pcl_msgs
  tf
  tf2_ros
  tf2_eigen
)

## 2. Trova le librerie esterne (Senza queste il linkaggio fallisce)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

## 3. Definizione del pacchetto Catkin
catkin_package(
  #INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros cv_bridge
)

## 4. Include directories (Dove cercare gli header .h e .hpp)
include_directories(
  #include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

# -------------------------------------------------------------------------
# 5. ESEGUIBILI (TARGETS)
# -------------------------------------------------------------------------

## Nodo 1: Il Combinatore (Genera la PointCloud da CoppeliaSim)
# ... (parte iniziale invariata)

# Nodo 1: Il Combinatore
add_executable(coppelia_combiner src/coppelia_combiner.cpp)
target_link_libraries(coppelia_combiner ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

# Nodo 2: Il Voxel
add_executable(voxel src/voxel.cpp)
target_link_libraries(voxel ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#Nodo 3: Filtro arm_segmentation https://www.youtube.com/watch?v=_FbbCxYjvoc&t=839s
add_executable(arm_segmentation src/arm_segmentation.cpp)
target_link_libraries(arm_segmentation ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#Nodo 4: Filtro arm_segmentation visuale 2
add_executable(arm_segmentation2 src/arm_segmentation2.cpp)
target_link_libraries(arm_segmentation2 ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#Nodo 5: Filtro arm_segmentation 3
add_executable(arm_segmentation3 src/arm_segmentation3.cpp)
target_link_libraries(arm_segmentation3 ${catkin_LIBRARIES} ${PCL_LIBRARIES})

