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

#include <rclcpp/rclcpp.hpp>

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <theora_wrappers/publisher.hpp>

#include <vision_msgs/msg/detection2_d.hpp>

#define UNUSED(arg) (void)(arg)
#define LINE std::cout << __FUNCTION__ << ", LINE: " << __LINE__ << std::endl;

using namespace geometry_msgs::msg;
using namespace rcl_interfaces::msg;
using namespace sensor_msgs::msg;
using namespace std_msgs::msg;
using namespace std_srvs::srv;
using namespace vision_msgs::msg;

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
            std::vector<int64_t>& objects_ids_);
  std::vector<Detection> run_inference(cv::Mat &input);

private:
  void load_onnx_network();

  cv::dnn::Net net;
  std::string model_path;
  std::string classes_path;
  bool cuda_enabled;
  std::vector<int64_t> objects_ids;

  std::vector<cv::Scalar> colors;

  std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

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

  /* image_transport subscriptions */
  std::shared_ptr<image_transport::CameraSubscriber> camera_sub_;

  /* Topic subscriptions callbacks */
  void camera_callback(const Image::ConstSharedPtr & msg,
                       const CameraInfo::ConstSharedPtr & camera_info_msg);

  /* Topic publishers */
  rclcpp::Publisher<Detection2D>::SharedPtr detections_pub_;

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
  cv::Mat camera_frame_, new_frame_;
  std_msgs::msg::Header last_header_;

  /* Internal state variables */
  cv::Mat cameraMatrix, distCoeffs, objPoints;
  Inference detector;

  /* Node parameters */
  bool always_pub_stream_ = false;
  bool autostart_ = false;
  bool best_effort_sub_qos_ = false;
  std::vector<int64_t> model_shape_ = {};
  int64_t image_sub_depth_ = 0;
  double model_score_threshold_ = 0.0;
  double model_NMS_threshold_ = 0.0;
  std::vector<int64_t> objects_ids_ = {};
  std::string onnx_path_ = "";
  std::string transport_ = "";
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
