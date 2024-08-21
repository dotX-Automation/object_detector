/**
 * Object Detector node definition.
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

#ifndef OBJECT_DETECTOR__OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR__OBJECT_DETECTOR_HPP

#include <algorithm>
#include <atomic>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <semaphore.h>

#include <dua_node/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <rclcpp/rclcpp.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber.hpp>
#include <image_transport/subscriber_filter.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <theora_wrappers/publisher.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <std_srvs/srv/set_bool.hpp>

#include <object_detector/inference.hpp>

using namespace geometry_msgs::msg;
using namespace sensor_msgs::msg;
using namespace std_msgs::msg;
using namespace vision_msgs::msg;

using namespace std_srvs::srv;

typedef message_filters::sync_policies::ExactTime<Image, Image> depth_sync_policy;

namespace object_detector
{

/**
 * Object detection node.
 */
class ObjectDetector : public dua_node::NodeBase
{
public:
  ObjectDetector(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());
  ~ObjectDetector();

private:
  /* Node initialization routines. */
  void init_inference();
  void init_parameters();
  void init_publishers();
  void init_services();
  void init_subscriptions();

  /* Synchronizer. */
  std::shared_ptr<message_filters::Synchronizer<depth_sync_policy>> sync_;

  /* Subscriptions. */
  std::shared_ptr<image_transport::SubscriberFilter> image_sub_sync_;
  std::shared_ptr<image_transport::SubscriberFilter> depth_sub_sync_;
  std::shared_ptr<image_transport::Subscriber> image_sub_;
  std::shared_ptr<image_transport::Subscriber> depth_sub_;

  /* Topic subscriptions callbacks. */
  void image_callback(const Image::ConstSharedPtr & image_msg);
  void sync_callback(
    const Image::ConstSharedPtr & image_msg,
    const Image::ConstSharedPtr & depth_msg);

  /* Topic publishers. */
  rclcpp::Publisher<Detection2DArray>::SharedPtr detections_pub_;

  /* Theora stream publishers. */
  std::shared_ptr<TheoraWrappers::Publisher> stream_pub_;

  /* Service servers. */
  rclcpp::Service<SetBool>::SharedPtr enable_server_;

  /* Service callbacks. */
  void enable_callback(
    SetBool::Request::SharedPtr req,
    SetBool::Response::SharedPtr resp);

  /* Data buffers. */
  cv::Mat camera_frame_{}, new_frame_{}, new_depth_{};
  std_msgs::msg::Header last_header_{};

  /* Detection engines. */
  Inference detector_;

  /* Node parameters. */
  bool always_publish_stream_ = false;
  bool autostart_ = false;
  std::vector<std::string> classes_ = {};
  std::vector<std::string> classes_targets_ = {};
  double model_NMS_threshold_ = 0.0;
  std::string model_path_ = "";
  double model_score_threshold_ = 0.0;
  std::vector<int64_t> model_shape_ = {};
  bool use_coco_classes_ = false;
  bool use_depth_ = false;
  bool use_gpu_ = false;
  bool verbose_ = false;
  int64_t worker_cpu_ = 0;

  /* Synchronization primitives for internal update operations. */
  std::atomic<bool> running_{false};
  sem_t sem1_, sem2_;

  /* Threads. */
  std::thread worker_;
  void worker_thread_routine();

  /* Utility routines. */
  void activate_detector();
  void deactivate_detector();
  Image::SharedPtr frame_to_msg(cv::Mat & frame);
};

} // namespace object_detector

#endif // OBJECT_DETECTOR__OBJECT_DETECTOR_HPP
