/**
 * Object Detector node implementation.
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

#include <string>
#include <stdexcept>

#include <object_detector/object_detector.hpp>

namespace ObjectDetector
{

/**
 * @brief Builds a new Object Detector node.
 *
 * @param opts Node options.
 *
 * @throws RuntimeError if initialization fails.
 */
ObjectDetectorNode::ObjectDetectorNode(const rclcpp::NodeOptions & node_options)
: NodeBase("object_detector", node_options, true)
{
  init_parameters();
  init_publishers();
  init_subscriptions();
  init_services();
  init_inference();

  RCLCPP_INFO(this->get_logger(), "Node initialized");
}

/**
 * @brief Finalizes node operation.
 */
ObjectDetectorNode::~ObjectDetectorNode()
{
  if (running_.load(std::memory_order_acquire)) {
    // Join worker thread
    running_.store(false, std::memory_order_release);
    sem_post(&sem1_);
    sem_post(&sem2_);
    worker_.join();

    // Shut down image and depth subscriber
    image_sub_.reset();
    image_sub_->unsubscribe();
    depth_sub_.reset();
    depth_sub_->unsubscribe();

    // Destroy semaphores
    sem_destroy(&sem1_);
    sem_destroy(&sem2_);
  }
  stream_pub_.reset();
}

/**
 * @brief Routine to initialize topic subscriptions.
 */
void ObjectDetectorNode::init_inference()
{
  if (use_coco_classes_)
    classes_ = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

  detector = Inference(onnx_path_,
                       cv::Size(model_shape_[0], model_shape_[1]),
                       use_gpu_,
                       &model_score_threshold_,
                       &model_NMS_threshold_,
                       classes_,
                       classes_targets_);
}

/**
 * @brief Routine to initialize topic subscriptions.
 */
void ObjectDetectorNode::init_subscriptions()
{
  if (autostart_) {
    // Initialize semaphores
    sem_init(&sem1_, 0, 1);
    sem_init(&sem2_, 0, 0);

    // Spawn worker thread
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(
      &ObjectDetectorNode::worker_thread_routine,
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
        char* err_msg = strerror_r(errno, err_msg_buf, 100);
        throw std::runtime_error(
          "ObjectDetectorNode::init_subscriptions: Failed to configure worker thread: " +
          std::string(err_msg));
      }
    }

    image_sub_ = std::make_shared<image_transport::SubscriberFilter>();
    depth_sub_ = std::make_shared<image_transport::SubscriberFilter>();

    // Subscribe to image topic
    image_sub_->subscribe(
      this,
      "/image",
      transport_,
      best_effort_sub_qos_ ?
      dua_qos::BestEffort::get_image_qos(image_sub_depth_).get_rmw_qos_profile() :
      dua_qos::Reliable::get_image_qos(image_sub_depth_).get_rmw_qos_profile());

    // Subscribe to depth topic
    depth_sub_->subscribe(
      this,
      "/depth",
      transport_,
      best_effort_sub_qos_ ?
      dua_qos::BestEffort::get_image_qos(image_sub_depth_).get_rmw_qos_profile() :
      dua_qos::Reliable::get_image_qos(image_sub_depth_).get_rmw_qos_profile());

    // Initialize synchronizer
    sync_ = std::make_shared<message_filters::Synchronizer<depth_sync_policy>>(
      depth_sync_policy(10),
      *image_sub_,
      *depth_sub_);
    sync_->registerCallback(&ObjectDetectorNode::sync_callback, this);
  }
}

/**
 * @brief Routine to initialize topic publishers.
 */
void ObjectDetectorNode::init_publishers()
{
  // Detections
  detections_pub_ = this->create_publisher<Detection2DArray>(
    "/detections",
    dua_qos::Reliable::get_datum_qos());

  // Detection stream
  stream_pub_ = std::make_shared<TheoraWrappers::Publisher>(
    this,
    "/detections_stream",
    dua_qos::BestEffort::get_image_qos(image_sub_depth_).get_rmw_qos_profile());
}

/**
 * @brief Routine to initialize service servers.
 */
void ObjectDetectorNode::init_services()
{
  // Enable
  enable_server_ = this->create_service<SetBool>(
    "~/enable",
    std::bind(
      &ObjectDetectorNode::enable_callback,
      this,
      std::placeholders::_1,
      std::placeholders::_2));
}

/**
 * @brief Worker routine.
 */
void ObjectDetectorNode::worker_thread_routine()
{
  while (true) {
    // Get new data
    std_msgs::msg::Header header;
    cv::Mat image{};
    sem_wait(&sem2_);
    if (!running_.load(std::memory_order_acquire)) { break; }
    image = new_frame_.clone();
    header = last_header_;
    sem_post(&sem1_);

    // Detect targets
    std::vector<Detection> output = detector.run_inference(image);
    int detections = output.size();

    // Return if no target is detected
    if (detections == 0 && !always_pub_stream_) { continue; }

    if (detections != 0)
    {
      Detection2DArray detections_msg;
      detections_msg.set__header(header);

      for (int i = 0; i < detections; i++)
      {
        Detection detection = output[i];

        RCLCPP_INFO(this->get_logger(), "Detected %s at (%d, %d, %d, %d) with confidence %f",
                    detection.class_name.c_str(), detection.box.x, detection.box.y,
                    detection.box.width, detection.box.height, detection.confidence);

        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;
        cv::Mat mask = detection.mask;

        // Detection box
        cv::rectangle(image, box, color, 2);

        // Segmentation mask
        if (!mask.empty())
        {
          cv::resize(mask, mask, box.size());
          mask.convertTo(mask, CV_8UC3, 255);
          cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

          cv::Mat roi = image(box);
          cv::addWeighted(roi, 1.0, mask, 0.3, 0.0, roi);
        }

        // Detection message
        Detection2D detection_msg;

        // Set hypotesis
        ObjectHypothesisWithPose result;
        result.hypothesis.set__class_id(detection.class_name);
        result.hypothesis.set__score(detection.confidence);
        result.pose.covariance[0] = -1.0;     // TODO
        result.pose.pose.position.set__x(0);  // TODO
        result.pose.pose.position.set__y(0);  // TODO
        result.pose.pose.position.set__z(0);  // TODO

        detection_msg.results.push_back(result);

        // Set bounding box
        detection_msg.set__header(header);
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

} // namespace ObjectDetector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ObjectDetector::ObjectDetectorNode)