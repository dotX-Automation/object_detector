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

  // Initialize data buffers
  camera_frame_ = std::make_shared<cv::Mat>();
  new_frame_ = std::make_shared<cv::Mat>();
  new_distances_ = std::make_shared<cv::Mat>();
  last_header_ = std::make_shared<Header>();
  new_depth_map_ = std::make_shared<PointCloud2>();

  // Initialize sensors
  std::vector<std::string> image_topics =
    this->get_parameter("subscriber_base_topic_name").as_string_array();
  std::vector<bool> best_effort_qos =
    this->get_parameter("subscriber_best_effort_qos").as_bool_array();
  std::vector<int64_t> depths =
    this->get_parameter("subscriber_depth").as_integer_array();
  std::vector<std::string> transports =
    this->get_parameter("subscriber_transport").as_string_array();
  std::vector<bool> use_depth =
    this->get_parameter("use_depth").as_bool_array();
  std::vector<bool> use_distances =
    this->get_parameter("use_distances").as_bool_array();
  std::vector<std::string> depth_topics =
    this->get_parameter("detection_depth_topics").as_string_array();
  std::vector<std::string> distances_topics =
    this->get_parameter("detection_distances_topics").as_string_array();
  std::vector<std::string> detection_output_topics =
    this->get_parameter("detection_output_topics").as_string_array();
  std::vector<std::string> detection_visual_output_topics =
    this->get_parameter("detection_visual_output_topics").as_string_array();
  std::vector<std::string> detection_stream_topics =
    this->get_parameter("detection_stream_topics").as_string_array();

  for (std::size_t i = 0; i < image_topics.size(); i++) {
    std::shared_ptr<Sensor> new_sensor = std::make_shared<Sensor>();

    // Configure topics
    new_sensor->depth_topic = depth_topics[i];
    new_sensor->distances_topic = distances_topics[i];
    new_sensor->detection_output_topic = detection_output_topics[i];
    new_sensor->detection_visual_output_topic = detection_visual_output_topics[i];
    new_sensor->detection_stream_topic = detection_stream_topics[i];
    new_sensor->subscriber_base_topic_name = image_topics[i];

    // Configure subscriber settings
    new_sensor->subscriber_best_effort_qos = best_effort_qos[i];
    new_sensor->subscriber_depth = depths[i];
    new_sensor->subscriber_transport = transports[i];

    // Configure sensor settings enforcing coherence
    if (use_depth[i]) {
      new_sensor->use_depth = true;
    } else if (use_distances[i]) {
      new_sensor->use_distances = true;
    }

    // Configure pointers to global data buffers
    new_sensor->curr_sensor_ptr = &curr_sensor_;
    new_sensor->camera_frame = camera_frame_;
    new_sensor->new_frame = new_frame_;
    new_sensor->new_distances = new_distances_;
    new_sensor->last_header = last_header_;
    new_sensor->new_depth_map = new_depth_map_;
    new_sensor->node_ptr = this;

    // Activate publishers
    new_sensor->detections_pub = this->create_publisher<Detection2DArray>(
      new_sensor->detection_output_topic,
      dua_qos::Reliable::get_datum_qos());
    new_sensor->visual_targets_pub = this->create_publisher<VisualTargets>(
      new_sensor->detection_visual_output_topic,
      dua_qos::Reliable::get_datum_qos());
    new_sensor->stream_pub = std::make_shared<TheoraWrappers::Publisher>(
      this,
      new_sensor->detection_stream_topic,
      dua_qos::BestEffort::get_image_qos().get_rmw_qos_profile());

    sensors_.push_back(new_sensor);
  }

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

  // Destroy sensors
  for (auto & sensor : sensors_) {
    sensor.reset();
  }

  // Destroy data buffers
  camera_frame_.reset();
  new_frame_.reset();
  new_distances_.reset();
  last_header_.reset();
  new_depth_map_.reset();
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

} // namespace object_detector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(object_detector::ObjectDetector)
