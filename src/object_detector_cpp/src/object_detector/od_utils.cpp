/**
 * Object Detector auxiliary functions.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 *
 * June 4, 2024
 */

/**
 * Copyright 2024 dotX Automation s.r.l.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pthread.h>
#include <sched.h>

#include <object_detector/object_detector.hpp>

namespace object_detector
{

/**
 * @brief Converts a frame into an Image message.
 *
 * @param frame cv::Mat storing the frame.
 * @return Shared pointer to a new Image message.
 */
Image::SharedPtr ObjectDetector::frame_to_msg(cv::Mat & frame)
{
  auto ros_image = std::make_shared<Image>();

  // Set frame-relevant image contents
  ros_image->set__width(frame.cols);
  ros_image->set__height(frame.rows);
  ros_image->set__encoding(sensor_msgs::image_encodings::BGR8);
  ros_image->set__step(frame.cols * frame.elemSize());
  ros_image->set__is_bigendian(false);

  // Copy frame data (this avoids the obsolete cv_bridge)
  size_t size = ros_image->step * frame.rows;
  ros_image->data.resize(size);
  memcpy(ros_image->data.data(), frame.data, size);

  return ros_image;
}

/**
 * @brief Initializes and activates the detector.
 *
 * @throws std::runtime_error if initialization fails.
 */
void ObjectDetector::activate_detector()
{
  // Initialize semaphores
  sem_init(&sem1_, 0, 1);
  sem_init(&sem2_, 0, 0);

  // Spawn worker thread
  running_.store(true, std::memory_order_release);
  worker_ = std::thread(
    &ObjectDetector::worker_thread_routine,
    this);
  if (worker_cpu_ != -1) {
    cpu_set_t worker_cpu_set;
    CPU_ZERO(&worker_cpu_set);
    CPU_SET(worker_cpu_, &worker_cpu_set);
    if (pthread_setaffinity_np(
        worker_.native_handle(),
        sizeof(cpu_set_t),
        &worker_cpu_set))
    {
      char err_msg_buf[100] = {};
      char * err_msg = strerror_r(errno, err_msg_buf, 100);
      throw std::runtime_error(
              "ObjectDetector::activate_detector: Failed to configure worker thread: " +
              std::string(err_msg));
    }
  }

  // Activate sensors
  for (auto & sensor : sensors_) {
    std::string base_topic_name = sensor->subscriber_base_topic_name;
    std::string transport = sensor->subscriber_transport;
    bool best_effort_qos = sensor->subscriber_best_effort_qos;
    int64_t depth = sensor->subscriber_depth;
    std::string depth_topic = sensor->depth_topic;
    std::string distances_topic = sensor->distances_topic;

    sensor->sem1 = &sem1_;
    sensor->sem2 = &sem2_;

    if (sensor->use_distances) {
      // Initialize camera_info topic name and data
      size_t pos = base_topic_name.find_last_of("/");
      std::string camera_info_topic_name = base_topic_name.substr(0, pos) + "/camera_info";
      sensor->got_camera_info = false;

      // Initialize image_transport subscribers
      sensor->image_sub_sync = std::make_shared<image_transport::SubscriberFilter>();
      sensor->distances_sub_sync = std::make_shared<image_transport::SubscriberFilter>();

      // Subscribe to image topic
      sensor->image_sub_sync->subscribe(
        this,
        base_topic_name,
        transport,
        best_effort_qos ?
        dua_qos::BestEffort::get_image_qos(depth).get_rmw_qos_profile() :
        dua_qos::Reliable::get_image_qos(depth).get_rmw_qos_profile());

      // Subscribe to camera_info topic
      sensor->camera_info_sub_sync = std::make_shared<message_filters::Subscriber<CameraInfo>>(
        this,
        camera_info_topic_name,
        best_effort_qos ?
        dua_qos::BestEffort::get_datum_qos().get_rmw_qos_profile() :
        dua_qos::Reliable::get_datum_qos().get_rmw_qos_profile());

      // Subscribe to distances topic
      sensor->distances_sub_sync->subscribe(
        this,
        distances_topic,
        "raw",
        dua_qos::Reliable::get_image_qos(depth).get_rmw_qos_profile());

      // Initialize synchronizer
      sensor->distances_sync = std::make_shared<message_filters::Synchronizer<distances_sync_policy>>(
        distances_sync_policy(depth),
        *(sensor->image_sub_sync),
        *(sensor->camera_info_sub_sync),
        *(sensor->distances_sub_sync));
      sensor->distances_sync->registerCallback(&Sensor::distances_sync_callback, sensor.get());

      sensor->is_rectified = base_topic_name.find("rect") != std::string::npos;
    } else if (sensor->use_depth) {
      // Initialize image_transport subscribers
      sensor->image_sub_sync = std::make_shared<image_transport::SubscriberFilter>();

      // Subscribe to image topic
      sensor->image_sub_sync->subscribe(
        this,
        base_topic_name,
        transport,
        best_effort_qos ?
        dua_qos::BestEffort::get_image_qos(depth).get_rmw_qos_profile() :
        dua_qos::Reliable::get_image_qos(depth).get_rmw_qos_profile());

      // Subscribe to depth topic
      sensor->depth_map_sub_sync = std::make_shared<message_filters::Subscriber<PointCloud2>>(
        this,
        depth_topic,
        best_effort_qos ?
        dua_qos::BestEffort::get_scan_qos(depth).get_rmw_qos_profile() :
        dua_qos::Reliable::get_scan_qos(depth).get_rmw_qos_profile());

      // Initialize synchronizer
      sensor->depth_sync = std::make_shared<message_filters::Synchronizer<depth_sync_policy>>(
        depth_sync_policy(depth),
        *(sensor->image_sub_sync),
        *(sensor->depth_map_sub_sync));
      sensor->depth_sync->registerCallback(&Sensor::depth_sync_callback, sensor.get());
    } else {
      sensor->image_sub = std::make_shared<image_transport::Subscriber>(
        image_transport::create_subscription(
          this,
          base_topic_name,
          std::bind(
            &Sensor::image_callback,
            sensor.get(),
            std::placeholders::_1),
          transport,
          best_effort_qos ?
          dua_qos::BestEffort::get_image_qos(depth).get_rmw_qos_profile() :
          dua_qos::Reliable::get_image_qos(depth).get_rmw_qos_profile()));
    }
  }

  RCLCPP_WARN(this->get_logger(), "Object Detector ACTIVATED");
}

/**
 * @brief Deactivates the detector.
 */
void ObjectDetector::deactivate_detector()
{
  // Join worker thread
  running_.store(false, std::memory_order_release);
  sem_post(&sem1_);
  sem_post(&sem2_);
  worker_.join();

  // Shutdown subscriptions and cleanup data
  for (auto & sensor : sensors_) {
    if (sensor->use_distances) {
      sensor->distances_sync.reset();
      sensor->image_sub_sync.reset();
      sensor->distances_sub_sync.reset();
      sensor->camera_info_sub_sync.reset();
    } else if (sensor->use_depth) {
      sensor->depth_sync.reset();
      sensor->image_sub_sync.reset();
      sensor->depth_map_sub_sync.reset();
    } else {
      sensor->image_sub.reset();
    }
  }

  // Destroy semaphores
  sem_destroy(&sem1_);
  sem_destroy(&sem2_);
}

} // namespace object_detector
