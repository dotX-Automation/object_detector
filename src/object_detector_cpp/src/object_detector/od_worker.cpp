/**
 * Object Detector worker thread routine.
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

#include <cmath>
#include <string>

#include <object_detector/object_detector.hpp>

namespace object_detector
{

/**
 * @brief Worker routine.
 */
void ObjectDetector::worker_thread_routine()
{
  while (true) {
    // Get new data
    Header header;
    cv::Mat image{}, distances{};
    PointCloud2 depth_map{};
    Sensor * sensor = nullptr;
    sem_wait(&sem2_);
    if (!running_.load(std::memory_order_acquire)) {
      break;
    }
    header = *last_header_;
    image = (*new_frame_).clone();
    distances = (*new_distances_).clone();
    depth_map = *new_depth_map_;
    sensor = curr_sensor_;
    sem_post(&sem1_);

    // Detect targets
    std::vector<Detection> output = detector_.run_inference(
      image,
      this->get_parameter("classes_targets").as_string_array());
    int detections = output.size();

    // Return if no target is detected
    if (detections > 0 && !always_publish_stream_) {
      continue;
    }

    if (detections != 0) {
      if (!sensor->use_distances || sensor->got_camera_info) {
        // I.e. if we are not using distances or we are and we have the camera info
        Detection2DArray detections_msg{};
        detections_msg.set__header(header);

        for (int i = 0; i < detections; i++) {
          Detection detection = output[i];

          if (verbose_) {
            RCLCPP_INFO(
              this->get_logger(), "Detected %s at (%d, %d, [%d, %d]) with confidence %f",
              detection.class_name.c_str(),
              detection.box.x,
              detection.box.y,
              detection.box.width,
              detection.box.height,
              detection.confidence);
          }

          cv::Rect box = detection.box;
          cv::Scalar color = detection.color;
          cv::Mat mask = detection.mask;
          cv::Point2f mask_centroid = detection.mask_centroid;

          float box_cx = box.x + box.width / 2.0f;
          float box_cy = box.y + box.height / 2.0f;

          // Draw detection box
          cv::rectangle(image, box, color, 2);

          // Prepare detection message
          Detection2D detection_msg{};
          detection_msg.set__header(header);

          // Fill hypotesis message
          ObjectHypothesisWithPose result{};
          result.hypothesis.set__class_id(detection.class_name);
          result.hypothesis.set__score(detection.confidence);

          // Compute and set target centroid
          double u;
          double v;
          if (mask.empty()) {
            // Set bounding box center as rectangle center
            u = box_cx;
            v = box_cy;
            detection_msg.bbox.center.position.set__x(box_cx);
            detection_msg.bbox.center.position.set__y(box_cy);
          } else {
            // Set bounding box center as mask centroid
            u = mask_centroid.x;
            v = mask_centroid.y;
            detection_msg.bbox.center.position.set__x(mask_centroid.x);
            detection_msg.bbox.center.position.set__y(mask_centroid.y);
          }

          if (sensor->use_distances) {
            // Get camera intrinsic parameters
            double fx = 0.0;
            double fy = 0.0;
            double cx = 0.0;
            double cy = 0.0;
            if (!sensor->is_rectified) {
              fx = sensor->camera_info.k[0];
              fy = sensor->camera_info.k[4];
              cx = sensor->camera_info.k[2];
              cy = sensor->camera_info.k[5];
            } else {
              fx = sensor->camera_info.p[0];
              fy = sensor->camera_info.p[5];
              cx = sensor->camera_info.p[2];
              cy = sensor->camera_info.p[6];
            }

            // Get bounding box rectangle from distances image
            cv::Mat distances_roi = distances(box);

            double sum;
            int count;

            // Compute distance between object and camera as the mean of the distances values in a ROI
            if (mask.empty()) {
              // No mask provided: ROI is the whole bounding box
              if (verbose_) {
                std::cout << "Mask not available" << std::endl;
              }

              count = cv::countNonZero(distances_roi);
              sum = cv::sum(distances_roi)[0];

              // Draw bbox centroid
              cv::circle(image, cv::Point2f(box_cx, box_cy), 5, cv::Scalar(0, 0, 255), -1);
            } else {
              // Mask provided: ROI is the intersection between the bounding box and the mask
              if (verbose_) {
                std::cout << "Mask available" << std::endl;
              }

              distances_roi.copyTo(distances_roi, mask);
              count = cv::countNonZero(distances_roi);
              sum = cv::sum(distances_roi)[0];

              // Draw segmentation mask and centroid
              cv::resize(mask, mask, box.size());
              mask.convertTo(mask, CV_8UC3, 255);
              cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

              cv::Mat roi = image(box);
              cv::addWeighted(roi, 1.0, mask, 0.3, 0.0, roi);

              cv::circle(image, mask_centroid, 5, cv::Scalar(0, 0, 255), -1);
            }
            if (count == 0) {
              continue;
            }
            double mean = sum / double(count);

            if (verbose_) {
              std::cout << "sum: " << sum << std::endl;
              std::cout << "count: " << count << std::endl;
              std::cout << "mean: " << mean << std::endl;
              std::cout << "x1: " << box.x << ", y1: " << box.y << std::endl;
              std::cout << "u: " << u << ", v: " << v << std::endl;
            }

            /**
             * Compute 3D coordinates of the centroid in the camera frame using camera intrinsic parameters
             * Rationale: we have camera intrinsic parameters and centroid image coordinates
             * but we don't have "depth", i.e. Z coordinate of the centroid in the camera frame.
             * Instead, we have the distance from the camera, so we can compute Z using that and changing variables.
             * Then, we can compute X and Y using Z and all the other data.
             * Computation is split into multiple steps for clarity.
             */
            double add_x_num = (u - cx) * (u - cx);
            double add_x_den = fx * fx;
            double add_y_num = (v - cy) * (v - cy);
            double add_y_den = fy * fy;
            double add_x = add_x_num / add_x_den;
            double add_y = add_y_num / add_y_den;
            double Z = mean / sqrt(add_x + add_y + 1);
            double X = (u - cx) * Z / fx;
            double Y = (v - cy) * Z / fy;

            if (verbose_) {
              std::cout << "X: " << X << std::endl;
              std::cout << "Y: " << Y << std::endl;
              std::cout << "Z: " << Z << std::endl;
            }

            result.pose.pose.position.set__x(X);
            result.pose.pose.position.set__y(Y);
            result.pose.pose.position.set__z(Z);
          } else if (sensor->use_depth) {
            if (!mask.empty()) {
              // Draw segmentation mask and centroid
              cv::resize(mask, mask, box.size());
              mask.convertTo(mask, CV_8UC3, 255);
              cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

              cv::Mat roi = image(box);
              cv::addWeighted(roi, 1.0, mask, 0.3, 0.0, roi);

              cv::circle(image, mask_centroid, 5, cv::Scalar(0, 0, 255), -1);
            } else {
              // Draw bbox centroid
              cv::circle(image, cv::Point2f(box_cx, box_cy), 5, cv::Scalar(0, 0, 255), -1);
            }

            // Get to the centroid in the depth map
            sensor_msgs::PointCloud2Iterator<float> iter_x(depth_map, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(depth_map, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(depth_map, "z");
            int to_iterate = v * depth_map.width + u;
            for (int i = 0; i < to_iterate; i++) {
              ++iter_x;
              ++iter_y;
              ++iter_z;
            }

            // Get centroid world coordinates from depth map
            double X(*iter_x);
            double Y(*iter_y);
            double Z(*iter_z);

            if (verbose_) {
              std::cout << "X: " << X << std::endl;
              std::cout << "Y: " << Y << std::endl;
              std::cout << "Z: " << Z << std::endl;
              std::cout << "u: " << u << ", v: " << v << std::endl;
            }

            // Discard this sample if coordinates are invalid (assumes a filled depth map)
            if (X == 0.0 && Y == 0.0 && Z == 0.0) {
              continue;
            }
            result.pose.pose.position.set__x(X);
            result.pose.pose.position.set__y(Y);
            result.pose.pose.position.set__z(Z);
          } else {
            result.pose.covariance[0] = -1.0;

            if (!mask.empty()) {
              // Draw segmentation mask and centroid
              cv::resize(mask, mask, box.size());
              mask.convertTo(mask, CV_8UC3, 255);
              cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

              cv::Mat roi = image(box);
              cv::addWeighted(roi, 1.0, mask, 0.3, 0.0, roi);

              cv::circle(image, mask_centroid, 5, cv::Scalar(0, 0, 255), -1);
            } else {
              // Draw bbox centroid
              cv::circle(image, cv::Point2f(box_cx, box_cy), 5, cv::Scalar(0, 0, 255), -1);
            }
          }

          detection_msg.results.push_back(result);

          detection_msg.bbox.set__size_x(detection.box.width);
          detection_msg.bbox.set__size_y(detection.box.height);

          detections_msg.detections.push_back(detection_msg);
        }

        // Publish detections
        sensor->detections_pub->publish(detections_msg);

        // Publish visual targets
        VisualTargets visual_targets_msg{};
        visual_targets_msg.set__targets(detections_msg);
        visual_targets_msg.set__image(*frame_to_msg(image));
        sensor->visual_targets_pub->publish(visual_targets_msg);
      }
    }

    // Publish processed image
    Image::SharedPtr processed_image_msg = frame_to_msg(image);
    processed_image_msg->set__header(header);
    sensor->stream_pub->publish(processed_image_msg);
  }

  RCLCPP_WARN(this->get_logger(), "Object Detector DEACTIVATED");
}

} // namespace object_detector
