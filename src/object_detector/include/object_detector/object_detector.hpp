/**
 * Object Detector node definition.
 *
 * Lorenzo Bianchi <lnz.bnc@gmail.com>
 * Intelligent Systems Lab <isl.torvergata@gmail.com>
 *
 * June 4, 2024
 */

/**
 * This is free software.
 * You can redistribute it and/or modify this file under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 3 of the License, or (at your option) any later
 * version.
 *
 * This file is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this file; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <algorithm>
#include <atomic>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <random>
#include <vector>

#include <semaphore.h>

#include <dua_node/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <rclcpp/rclcpp.hpp>

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber.hpp>
#include <image_transport/subscriber_filter.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <theora_wrappers/publisher.hpp>


#define UNUSED(arg) (void)(arg)
#define LINE std::cout << __FUNCTION__ << ", LINE: " << __LINE__ << std::endl;

using namespace geometry_msgs::msg;
using namespace rcl_interfaces::msg;
using namespace sensor_msgs::msg;
using namespace std_msgs::msg;
using namespace std_srvs::srv;
using namespace vision_msgs::msg;

typedef message_filters::sync_policies::ExactTime<Image, Image> depth_sync_policy;

namespace ObjectDetector
{

struct Detection
{
  cv::Rect box{};
  int class_id{0};
  std::string class_name{};
  cv::Scalar color{};
  float confidence{0.0};
  cv::Mat mask{};
};

class Inference
{
public:
  Inference() = default;
  Inference(std::string &onnx_model_path,
            cv::Size model_input_shape,
            bool run_with_cuda,
            double* score_threshold,
            double* nms_threshold,
            std::vector<std::string>& classes,
            std::vector<std::string>& classes_targets);
  std::vector<Detection> run_inference(cv::Mat &input);

private:
  void load_onnx_network();

  cv::dnn::Net net;
  std::string model_path;
  std::string classes_path;
  bool cuda_enabled;

  std::vector<std::string> classes;
  std::vector<std::string> classes_targets;

  std::vector<cv::Scalar> colors;

  cv::Size2f model_shape;

  double* score_threshold;
  double* nms_threshold;
};

/**
 * Object detection node.
 */
class ObjectDetectorNode : public dua_node::NodeBase
{
public:
  ObjectDetectorNode(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());
  ~ObjectDetectorNode();

private:
  /* Node initialization routines */
  void init_inference();
  void init_parameters();
  void init_publishers();
  void init_services();
  void init_subscriptions();

  /* Synchronizer */
  std::shared_ptr<message_filters::Synchronizer<depth_sync_policy>> sync_;

  /* Subscriptions */
  std::shared_ptr<image_transport::SubscriberFilter> image_sub_sync_;
  std::shared_ptr<image_transport::SubscriberFilter> depth_sub_sync_;
  std::shared_ptr<image_transport::Subscriber> image_sub_;
  std::shared_ptr<image_transport::Subscriber> depth_sub_;

  /* Topic subscriptions callbacks */
  void image_callback(const Image::ConstSharedPtr & image_msg);
  void sync_callback(const Image::ConstSharedPtr & image_msg,
                     const Image::ConstSharedPtr & depth_msg);

  /* Topic publishers */
  rclcpp::Publisher<Detection2DArray>::SharedPtr detections_pub_;

  /* Theora stream publishers. */
  std::shared_ptr<TheoraWrappers::Publisher> stream_pub_;

  /* Service servers callback groups */
  rclcpp::CallbackGroup::SharedPtr enable_cgroup_;

  /* Service servers */
  rclcpp::Service<SetBool>::SharedPtr enable_server_;

  /* Service callbacks */
  void enable_callback(
    SetBool::Request::SharedPtr req,
    SetBool::Response::SharedPtr resp);

  /* Data buffers */
  cv::Mat camera_frame_, new_frame_, new_depth_;
  std_msgs::msg::Header last_header_;

  /* Internal state variables */
  Inference detector;

  /* Node parameters */
  bool always_pub_stream_ = false;
  bool autostart_ = false;
  bool best_effort_sub_qos_ = false;
  std::vector<std::string> classes_ = {};
  std::vector<std::string> classes_targets_ = {};
  std::vector<int64_t> model_shape_ = {};
  int64_t image_sub_depth_ = 0;
  double model_score_threshold_ = 0.0;
  double model_NMS_threshold_ = 0.0;
  std::string onnx_path_ = "";
  std::string transport_ = "";
  bool use_coco_classes_ = false;
  bool use_depth_ = false;
  bool use_gpu_ = false;
  int64_t worker_cpu_ = 0;

  /* Synchronization primitives for internal update operations */
  std::atomic<bool> running_{false};
  sem_t sem1_, sem2_;

  /* Threads */
  std::thread worker_;

  /* Utility routines */
  void worker_thread_routine();
  Image::SharedPtr frame_to_msg(cv::Mat & frame);
};

} // namespace ObjectDetector

#endif // OBJECT_DETECTOR_HPP
