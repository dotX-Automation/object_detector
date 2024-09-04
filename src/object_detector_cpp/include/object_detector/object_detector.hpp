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
#include <stdexcept>
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
#include <image_transport/subscriber_filter.hpp>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <theora_wrappers/publisher.hpp>

#include <dua_interfaces/msg/visual_targets.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <std_srvs/srv/set_bool.hpp>

#include <object_detector/inference.hpp>

using namespace dua_interfaces::msg;
using namespace geometry_msgs::msg;
using namespace sensor_msgs::msg;
using namespace std_msgs::msg;
using namespace vision_msgs::msg;

using namespace std_srvs::srv;

typedef message_filters::sync_policies::ExactTime<Image, CameraInfo, Image> distances_sync_policy;
typedef message_filters::sync_policies::ExactTime<Image, PointCloud2> depth_sync_policy;

namespace object_detector
{

/**
 * Embeds data of a sensor used in batch mode.
 */
class Sensor
{
public:
  Sensor();
  ~Sensor();

  /* Topic names. */
  std::string depth_topic = "";
  std::string distances_topic = "";
  std::string detection_output_topic = "";
  std::string detection_visual_output_topic = "";
  std::string detection_stream_topic = "";
  std::string subscriber_base_topic_name = "";

  /* Synchronizers. */
  std::shared_ptr<message_filters::Synchronizer<distances_sync_policy>> distances_sync;
  std::shared_ptr<message_filters::Synchronizer<depth_sync_policy>> depth_sync;

  /* Topic subscriptions. */
  std::shared_ptr<image_transport::SubscriberFilter> image_sub_sync;
  std::shared_ptr<image_transport::SubscriberFilter> distances_sub_sync;
  std::shared_ptr<message_filters::Subscriber<CameraInfo>> camera_info_sub_sync;
  std::shared_ptr<message_filters::Subscriber<PointCloud2>> depth_map_sub_sync;
  std::shared_ptr<image_transport::Subscriber> image_sub;

  /* Topic subscriptions callbacks. */
  void image_callback(const Image::ConstSharedPtr & image_msg);
  void distances_sync_callback(
    const Image::ConstSharedPtr & image_msg,
    const CameraInfo::ConstSharedPtr & camera_info_msg,
    const Image::ConstSharedPtr & distances_msg);
  void depth_sync_callback(
    const Image::ConstSharedPtr & image_msg,
    const PointCloud2::ConstSharedPtr & depth_msg);

  /* Topic publishers. */
  rclcpp::Publisher<Detection2DArray>::SharedPtr detections_pub;
  rclcpp::Publisher<VisualTargets>::SharedPtr visual_targets_pub;

  /* Theora stream publishers. */
  std::shared_ptr<TheoraWrappers::Publisher> stream_pub;

  /* Data buffers and state variables. */
  bool subscriber_best_effort_qos = false;
  bool use_depth = false;
  bool use_distances = false;
  int64_t subscriber_depth = 0L;
  std::string subscriber_transport = "";
  bool is_rectified = false;
  bool got_camera_info = false;
  CameraInfo camera_info{};
  rclcpp::Node * node_ptr = nullptr;

  /* Pointers to global data buffers and their synchronization primitives. */
  Sensor ** curr_sensor_ptr = nullptr;
  std::shared_ptr<cv::Mat> camera_frame;
  std::shared_ptr<cv::Mat> new_frame;
  std::shared_ptr<cv::Mat> new_distances;
  Header::SharedPtr last_header;
  PointCloud2::SharedPtr new_depth_map;
  sem_t * sem1 = nullptr;
  sem_t * sem2 = nullptr;
};

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
  void init_services();
  void init_subscriptions();

  /* Service servers. */
  rclcpp::Service<SetBool>::SharedPtr enable_server_;

  /* Service callbacks. */
  void enable_callback(
    SetBool::Request::SharedPtr req,
    SetBool::Response::SharedPtr resp);

  /* Data buffers. */
  std::shared_ptr<cv::Mat> camera_frame_, new_frame_, new_distances_;
  Header::SharedPtr last_header_;
  PointCloud2::SharedPtr new_depth_map_;

  /* Sensors. */
  std::vector<std::shared_ptr<Sensor>> sensors_;
  Sensor * curr_sensor_ = nullptr;

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
