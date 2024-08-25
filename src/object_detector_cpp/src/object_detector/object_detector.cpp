/**
 * Object Detector node implementation.
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
 * @brief Constructor.
 *
 * @param opts Node options.
 *
 * @throws std::runtime_error if parameters are not coherent.
 */
ObjectDetector::ObjectDetector(const rclcpp::NodeOptions & node_options)
: NodeBase("object_detector", node_options, true)
{
  init_parameters();
  init_inference();
  init_publishers();
  init_subscriptions();
  init_services();

  // Check that parameters are coherently set
  if (use_distances_ && use_depth_) {
    RCLCPP_FATAL(this->get_logger(), "Cannot use both distances and depth at the same time");
    throw std::runtime_error("Cannot use both distances and depth at the same time");
  }

  RCLCPP_INFO(this->get_logger(), "Node initialized");
}

/**
 * @brief Destructor.
 */
ObjectDetector::~ObjectDetector()
{
  if (running_.load(std::memory_order_acquire)) {
    deactivate_detector();
  }
  stream_pub_.reset();
}

/**
 * @brief Routine to initialize inference engine.
 */
void ObjectDetector::init_inference()
{
  if (use_coco_classes_) {
    classes_ =
    {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
      "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
      "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
      "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
      "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
      "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
      "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
      "toothbrush"};
  }

  detector_ = Inference(
    model_path_,
    cv::Size(model_shape_[0], model_shape_[1]),
    use_gpu_,
    &model_score_threshold_,
    &model_NMS_threshold_,
    classes_,
    classes_targets_,
    verbose_);
}

/**
 * @brief Routine to initialize topic subscriptions.
 */
void ObjectDetector::init_subscriptions()
{
  if (autostart_) {
    activate_detector();
  }
}

/**
 * @brief Routine to initialize topic publishers.
 */
void ObjectDetector::init_publishers()
{
  // detections
  detections_pub_ = this->create_publisher<Detection2DArray>(
    "~/detections",
    dua_qos::Reliable::get_datum_qos());

  // detections_stream
  stream_pub_ = std::make_shared<TheoraWrappers::Publisher>(
    this,
    "~/detections_stream",
    dua_qos::BestEffort::get_image_qos().get_rmw_qos_profile());
}

/**
 * @brief Routine to initialize service servers.
 */
void ObjectDetector::init_services()
{
  // enable
  enable_server_ = this->create_service<SetBool>(
    "~/enable",
    std::bind(
      &ObjectDetector::enable_callback,
      this,
      std::placeholders::_1,
      std::placeholders::_2));
}

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
    sem_wait(&sem2_);
    if (!running_.load(std::memory_order_acquire)) {
      break;
    }
    header = last_header_;
    image = new_frame_.clone();
    distances = new_distances_.clone();
    depth_map = new_depth_map_;
    sem_post(&sem1_);

    // Detect targets
    std::vector<Detection> output = detector_.run_inference(image);
    int detections = output.size();

    // Return if no target is detected
    if (detections == 0 && !always_publish_stream_) {continue;}

    if (detections != 0) {
      if (!use_distances_ || got_camera_info_) {
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

          // Draw detection box
          cv::rectangle(image, box, color, 2);

          // Prepare detection message
          Detection2D detection_msg{};
          detection_msg.set__header(header);

          // Fill hypotesis message
          ObjectHypothesisWithPose result{};
          result.hypothesis.set__class_id(detection.class_name);
          result.hypothesis.set__score(detection.confidence);

          if (distances.data != nullptr) {
            // Get camera intrinsic parameters
            double fx = camera_info_.k[0];
            double fy = camera_info_.k[4];
            double cx = camera_info_.k[2];
            double cy = camera_info_.k[5];

            // Get bounding box rectangle from distances image
            cv::Mat distances_roi = distances(box);

            double sum;
            int count;

            // Compute distance between object and camera as the mean of the distances values in a ROI
            if (mask.empty()) {
              // No mask provided: ROI is the whole bounding box
              if (verbose_) {
                std::cout << "Mask empty" << std::endl;
              }

              count = cv::countNonZero(distances_roi);
              sum = cv::sum(distances_roi)[0];
            } else {
              // Mask provided: ROI is the intersection between the bounding box and the mask
              if (verbose_) {
                std::cout << "Mask not empty" << std::endl;
              }

              distances_roi.copyTo(distances_roi, mask);
              count = cv::countNonZero(distances_roi);
              sum = cv::sum(distances_roi)[0];

              // Draw segmentation mask
              cv::resize(mask, mask, box.size());
              mask.convertTo(mask, CV_8UC3, 255);
              cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

              cv::Mat roi = image(box);
              cv::addWeighted(roi, 1.0, mask, 0.3, 0.0, roi);
            }
            if (count == 0) {
              continue;
            }
            double mean = sum / double(count);

            // Compute centroid image coordinates
            double u = box.x + (box.width / 2.0);
            double v = box.y + (box.height / 2.0);

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
          } else if (use_depth_) {
            if (!mask.empty()) {
              // Draw segmentation mask
              cv::resize(mask, mask, box.size());
              mask.convertTo(mask, CV_8UC3, 255);
              cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

              cv::Mat roi = image(box);
              cv::addWeighted(roi, 1.0, mask, 0.3, 0.0, roi);
            }

            // Compute centroid image coordinates
            double u = box.x + (box.width / 2.0);
            double v = box.y + (box.height / 2.0);

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
          }
          detection_msg.results.push_back(result);

          // Set bounding box
          detection_msg.bbox.center.position.set__x(detection.box.x + detection.box.width / 2);
          detection_msg.bbox.center.position.set__y(detection.box.y + detection.box.height / 2);
          detection_msg.bbox.set__size_x(detection.box.width);
          detection_msg.bbox.set__size_y(detection.box.height);

          detections_msg.detections.push_back(detection_msg);
        }

        // Publish detections
        detections_pub_->publish(detections_msg);
      }
    }

    // Publish processed image
    camera_frame_ = image; // doesn't copy image data, but sets data type...

    // Create processed image message
    Image::SharedPtr processed_image_msg = frame_to_msg(camera_frame_);
    processed_image_msg->set__header(header);

    // Publish processed image
    stream_pub_->publish(processed_image_msg);
  }

  RCLCPP_WARN(this->get_logger(), "Object Detector DEACTIVATED");
}

} // namespace object_detector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(object_detector::ObjectDetector)
