cmake_minimum_required(VERSION 3.8)
project(object_detector_cpp)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(dua_app_management REQUIRED)
find_package(dua_interfaces REQUIRED)
find_package(dua_node REQUIRED)
find_package(dua_qos_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV 4.7.0 REQUIRED)
find_package(params_manager REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(theora_wrappers REQUIRED)
find_package(Threads REQUIRED)
find_package(vision_msgs REQUIRED)

# Generate parameters source code
generate_init_parameters(
  YAML_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/object_detector/params.yaml"
  OUT_FILE "init_parameters.cpp")

# Configure Object Detector node source files
file(GLOB OBJECT_DETECTOR_NODE_SRCS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "src/object_detector/*.cpp")
list(APPEND OBJECT_DETECTOR_NODE_SRCS init_parameters.cpp)

# Object Detector node
add_library(object_detector SHARED ${OBJECT_DETECTOR_NODE_SRCS})
target_compile_definitions(object_detector PRIVATE COMPOSITION_BUILDING_DLL)
target_include_directories(object_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(object_detector PUBLIC c_std_99 cxx_std_17) # Require C99 and C++17
target_link_libraries(object_detector
  ${OpenCV_LIBS}
  Threads::Threads)
ament_target_dependencies(object_detector
  dua_app_management
  dua_interfaces
  dua_node
  dua_qos_cpp
  geometry_msgs
  image_transport
  message_filters
  params_manager
  rclcpp_components
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  theora_wrappers
  vision_msgs)
rclcpp_components_register_nodes(object_detector "object_detector::ObjectDetector")

# Object Detector standalone application
add_executable(object_detector_app
  src/object_detector_app.cpp)
target_include_directories(object_detector_app PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(object_detector_app PUBLIC c_std_99 cxx_std_17) # Require C99 and C++17
target_link_libraries(object_detector_app object_detector)
ament_target_dependencies(object_detector_app
  dua_app_management
  rclcpp)

# Install Object Detector component
install(TARGETS object_detector
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Install Object Detector app
install(TARGETS object_detector_app
  DESTINATION lib/${PROJECT_NAME})

# Install Config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

# Install Launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
