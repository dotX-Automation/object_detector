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
 */
ObjectDetector::ObjectDetector(const rclcpp::NodeOptions & node_options)
: NodeBase("object_detector", node_options, true)
{
  init_parameters();
  init_inference();
  init_publishers();
  init_subscriptions();
  init_services();

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
    cv::Mat image{}, depth{};
    sem_wait(&sem2_);
    if (!running_.load(std::memory_order_acquire)) {
      break;
    }
    header = last_header_;
    image = new_frame_.clone();
    depth = new_depth_.clone();
    sem_post(&sem1_);

    // Detect targets
    std::vector<Detection> output = detector_.run_inference(image);
    int detections = output.size();

    // Return if no target is detected
    if (detections == 0 && !always_publish_stream_) {continue;}

    if (detections != 0) {
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

        // Draw segmentation mask
        if (!mask.empty()) {
          cv::resize(mask, mask, box.size());
          mask.convertTo(mask, CV_8UC3, 255);
          cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

          cv::Mat roi = image(box);
          cv::addWeighted(roi, 1.0, mask, 0.3, 0.0, roi);
        }

        // Prepare detection message
        Detection2D detection_msg{};
        detection_msg.set__header(header);

        // Fill hypotesis message
        ObjectHypothesisWithPose result{};
        result.hypothesis.set__class_id(detection.class_name);
        result.hypothesis.set__score(detection.confidence);

        if (depth.data != nullptr) {
          // Get bounding box rectangle from depth image
          cv::Mat depth_roi = depth(box);

          double sum = 0.0;
          int count = 0;

          if (mask.empty()) {
            if (verbose_) {
              std::cout << "Mask empty" << std::endl;
            }
            for (int i = 0; i < depth_roi.rows; i++) {
              for (int j = 0; j < depth_roi.cols; j++) {
                if (!std::isnan(depth_roi.at<float>(i, j))) {
                  sum += double(depth_roi.at<float>(i, j));
                  count++;
                }
              }
            }
          } else {
            if (verbose_) {
              std::cout << "Mask not empty" << std::endl;
            }
            for (int i = 0; i < depth_roi.rows; i++) {
              for (int j = 0; j < depth_roi.cols; j++) {
                if (!std::isnan(depth_roi.at<float>(i, j)) && mask.at<uchar>(i, j) > 0) {
                  sum += double(depth_roi.at<float>(i, j));
                  count++;
                }
              }
            }
          }
          if (count == 0) {
            continue;
          }
          double mean = sum / double(count);

          int x1 = box.x;
          int y1 = box.y;
          int x2 = box.x + box.width;
          int y2 = box.y + box.height;

          if (verbose_) {
            std::cout << "x1: " << x1 << ", y1: " << y1 << std::endl;
            std::cout << "x2: " << x2 << ", y2: " << y2 << std::endl;
            std::cout << "Image size: " << image.size().width << ", " << image.size().height <<
              std::endl;
          }

          double w = image.size().width;
          double h = image.size().height;
          double u = (x1 + x2 - w) / 2.0 / w;
          double v = (y1 + y2 - h) / 2.0 / h;

          if (verbose_) {
            std::cout << "u: " << u << ", v: " << v << std::endl;
            std::cout << "mean: " << mean << std::endl;
          }

          double Z = mean / sqrt(u * u + v * v + 1);
          double Y = Z * v;
          double X = Z * u;

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
