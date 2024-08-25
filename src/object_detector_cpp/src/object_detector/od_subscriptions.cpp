/**
 * Object Detector topic subscription callbacks.
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

#include <object_detector/object_detector.hpp>

namespace object_detector
{

/**
 * @brief Parses a new image message.
 *
 * @param msg Image message to parse.
 */
void ObjectDetector::image_callback(const Image::ConstSharedPtr & msg)
{
  // Convert image_msg to OpenCV image
  cv::Mat frame = cv::Mat(
    msg->height,
    msg->width,
    CV_8UC3,
    (void *)(msg->data.data()));

  // Pass data to worker thread
  sem_wait(&sem1_);
  new_frame_ = frame.clone();
  last_header_ = msg->header;
  sem_post(&sem2_);
}

/**
 * @brief Parses a new image with distances data.
 *
 * @param image_msg Image message to parse.
 * @param camera_info_msg Camera info message to parse.
 * @param distances_msg Depth image message to parse.
 */
void ObjectDetector::distances_sync_callback(
  const Image::ConstSharedPtr & image_msg,
  const CameraInfo::ConstSharedPtr & camera_info_msg,
  const Image::ConstSharedPtr & distances_msg)
{
  // Register camera_info data
  if (!got_camera_info_) {
    camera_info_ = *camera_info_msg;
    got_camera_info_ = true;
  }

  // Check that input data dimensions are consistent
  if (image_msg->width != distances_msg->width || image_msg->height != distances_msg->height) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Input data dimensions are inconsistent: image (%u, %u), distances (%u, %u)",
      image_msg->width, image_msg->height,
      distances_msg->width, distances_msg->height);
    return;
  }

  // Convert image_msg to OpenCV image
  cv::Mat frame = cv::Mat(
    image_msg->height,
    image_msg->width,
    CV_8UC3,
    (void *)(image_msg->data.data()));

  // Convert distances_msg to OpenCV image
  cv::Mat distances = cv::Mat(
    distances_msg->height,
    distances_msg->width,
    CV_64FC1,
    (void *)(distances_msg->data.data()));

  // Pass data to worker thread
  sem_wait(&sem1_);
  new_frame_ = frame.clone();
  new_distances_ = distances.clone();
  last_header_ = image_msg->header;
  sem_post(&sem2_);
}

/**
 * @brief Parses a new image with depth data.
 *
 * @param image_msg Image message to parse.
 * @param depth_msg Depth map message to parse.
 */
void ObjectDetector::depth_sync_callback(
  const Image::ConstSharedPtr & image_msg,
  const PointCloud2::ConstSharedPtr & depth_msg)
{
  // Check that input data dimensions are consistent
  if (image_msg->width != depth_msg->width || image_msg->height != depth_msg->height) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Input data dimensions are inconsistent: image (%u, %u), depth_map (%u, %u)",
      image_msg->width, image_msg->height,
      depth_msg->width, depth_msg->height);
    return;
  }

  // Convert image_msg to OpenCV image
  cv::Mat frame = cv::Mat(
    image_msg->height,
    image_msg->width,
    CV_8UC3,
    (void *)(image_msg->data.data()));

  // Pass data to worker thread
  sem_wait(&sem1_);
  new_frame_ = frame.clone();
  last_header_ = image_msg->header;
  new_depth_map_ = *depth_msg;
  sem_post(&sem2_);
}

} // namespace object_detector
