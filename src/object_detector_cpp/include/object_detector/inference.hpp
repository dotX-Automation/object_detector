/**
 * Object Detector inference modules.
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

#ifndef OBJECT_DETECTOR__INFERENCE_HPP
#define OBJECT_DETECTOR__INFERENCE_HPP

#include <algorithm>
#include <atomic>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace object_detector
{

/**
 * Holds data of a detection.
 */
struct Detection
{
  cv::Rect box{};
  int class_id{0};
  std::string class_name{};
  cv::Scalar color{};
  float confidence{0.0};
  cv::Mat mask{};
  cv::Point2f mask_centroid{};
};

/**
 * Runs inference on an ONNX model using OpenCV DNN module.
 */
class Inference
{
public:
  Inference() = default;
  Inference(
    std::string & onnx_model_path,
    cv::Size model_input_shape,
    bool run_with_cuda,
    double * score_threshold,
    double * nms_threshold,
    std::vector<std::string> & classes,
    bool verbose,
    int colors_seed);

  std::vector<Detection> run_inference(
    cv::Mat & input,
    const std::vector<std::string> & classes_targets);

private:
  void load_onnx_network();

  cv::dnn::Net net_;
  std::string model_path_;
  std::string classes_path_;
  bool cuda_enabled_;

  std::vector<std::string> classes_;

  std::vector<cv::Scalar> colors_;

  cv::Size2f model_shape_;

  double * score_threshold_;
  double * nms_threshold_;

  bool verbose_;
};

} // namespace object_detector

#endif // OBJECT_DETECTOR__INFERENCE_HPP
