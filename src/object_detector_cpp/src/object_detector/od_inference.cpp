/**
 * Object Detector inference implementation.
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

#include <object_detector/inference.hpp>

namespace object_detector
{

/**
 * @brief Inference class constructor.
 *
 * @param onnx_model_path string with model path.
 * @param model_input_shape cv::Size with model input shape.
 * @param run_with_cuda boolean to run with CUDA.
 * @param score_threshold pointer to score threshold value.
 * @param nms_threshold pointer to NMS threshold value.
 * @param classes vector of classes names.
 * @param classes_targets vector of classes names to be found.
 * @param verbose verbose flag.
 */
Inference::Inference(
  std::string & onnx_model_path,
  cv::Size model_input_shape,
  bool run_with_cuda,
  double * score_threshold,
  double * nms_threshold,
  std::vector<std::string> & classes,
  std::vector<std::string> & classes_targets,
  bool verbose)
{
  this->model_path_ = onnx_model_path;
  this->model_shape_ = model_input_shape;
  this->cuda_enabled_ = run_with_cuda;
  this->score_threshold_ = score_threshold;
  this->nms_threshold_ = nms_threshold;
  this->classes_ = classes;
  this->classes_targets_ = classes_targets;
  this->verbose_ = verbose;

  // Create colors vector randomly
  int seed = 1;
  std::mt19937 gen = std::mt19937(seed);
  std::uniform_int_distribution<int> dis = std::uniform_int_distribution<int>(100, 255);
  for (size_t i = 0; i < classes.size(); i++) {
    colors_.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
  }

  load_onnx_network();

  if (verbose_) {
    // Print classes names to be found
    std::cout << "Target classes: " << std::endl;
    for (std::string target : classes_targets_) {
      std::cout << "\t- " << target << std::endl;
    }
  }
}

/**
 * @brief Inference class main method to run inference on a given image.
 *
 * @param input cv::Mat with input image.
 * @return Vector of objects found in the image.
 */
std::vector<Detection> Inference::run_inference(cv::Mat & input)
{
  int cols = input.cols;
  int rows = input.rows;

  if (cols == 0 || rows == 0) {
    if (verbose_) std::cout << "Image size is 0" << std::endl;
    return {};
  }

  cv::Mat blob;
  cv::dnn::blobFromImage(input, blob, 1.0 / 255.0, model_shape_, cv::Scalar(), true, false);
  net_.setInput(blob);

  std::vector<cv::Mat> outputs;
  net_.forward(outputs, net_.getUnconnectedOutLayersNames());

  // Detection
  cv::Mat boxes_output = outputs[0];

  int boxes_rows = boxes_output.size[2];
  int boxes_dims = boxes_output.size[1];

  boxes_output = boxes_output.reshape(1, boxes_dims).t();

  float x_factor = static_cast<float>(cols) / model_shape_.width;
  float y_factor = static_cast<float>(rows) / model_shape_.height;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<int> indices;

  float * data = (float *)boxes_output.data;

  for (int i = 0; i < boxes_rows; i++) {
    cv::Mat scores(1, classes_.size(), CV_32FC1, data + 4);

    cv::Point class_id;
    double max_class_score;
    cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

    if (max_class_score > *score_threshold_) {
      indices.push_back(i);

      // Limit values between 0 and image size
      float cx = std::max(std::min(data[0], model_shape_.width), 0.0f);
      float cy = std::max(std::min(data[1], model_shape_.height), 0.0f);
      float w = std::max(std::min(data[2], model_shape_.width), 0.0f);
      float h = std::max(std::min(data[3], model_shape_.height), 0.0f);

      int left = int((cx - w / 2) * x_factor);
      int top = int((cy - h / 2) * y_factor);
      int width = int(w * x_factor);
      int height = int(h * y_factor);

      // Limit left, top, width, height values between 0 and image size
      left = std::max(std::min(left, cols), 0);
      top = std::max(std::min(top, rows), 0);
      if (left + width > cols) {
        width = cols - left;
      }
      if (top + height > rows) {
        height = rows - top;
      }

      boxes.push_back(cv::Rect(left, top, width, height));
      confidences.push_back(max_class_score);
      class_ids.push_back(class_id.x);
    }

    data += boxes_dims;
  }

  std::vector<int> nms_result;
  std::vector<int> nms_result_to_skip;
  cv::dnn::NMSBoxes(boxes, confidences, *score_threshold_, *nms_threshold_, nms_result);
  if (nms_result.empty()) {
    return {};
  }

  size_t nms_result_size = nms_result.size();

  std::vector<Detection> detections;
  for (size_t i = 0; i < nms_result.size(); i++) {
    int idx = nms_result[i];
    int class_id = class_ids[idx];
    std::string class_name = classes_[class_id];

    if (std::find(
        classes_targets_.begin(),
        classes_targets_.end(),
        class_name) == classes_targets_.end())
    {
      nms_result_to_skip.push_back(i);
      nms_result_size--;
      continue;
    }

    Detection result;
    result.box = boxes[idx];
    result.class_id = class_id;
    result.class_name = class_name;
    result.color = colors_[result.class_id];
    result.confidence = confidences[idx];

    detections.push_back(result);
  }

  if (nms_result_size == 0) {
    return detections;
  }

  // Segmentation
  if (outputs.size() > 1) {
    cv::Mat masks_output = outputs[1];
    int mask_proto = masks_output.size[1];
    int mask_height = masks_output.size[2];
    int mask_width = masks_output.size[3];

    if (mask_proto == 0 || mask_height == 0 || mask_width == 0) {
      if (verbose_) std::cout << "Mask size is 0" << std::endl;
      return detections;
    }

    float x_factor_segm = x_factor * model_shape_.width / static_cast<float>(mask_width);
    float y_factor_segm = y_factor * model_shape_.height / static_cast<float>(mask_height);

    data = (float *)boxes_output.data;
    cv::Mat masks_prediction(mask_proto, 0, CV_32FC1);
    for (size_t i = 0; i < nms_result.size(); i++) {
      if (std::find(
          nms_result_to_skip.begin(),
          nms_result_to_skip.end(),
          i) != nms_result_to_skip.end())
      {
        continue;
      }

      int idx = nms_result[i];
      cv::Mat masks_temp(
        mask_proto,
        1,
        CV_32FC1,
        data + indices[idx] * boxes_dims + 4 + classes_.size());
      cv::hconcat(masks_prediction, masks_temp, masks_prediction);
    }

    masks_output = masks_output.reshape(1, {mask_proto, mask_height * mask_width}).t();

    cv::gemm(masks_output, masks_prediction, 1, cv::noArray(), 0, masks_output);

    // Compute sigmoid of masks_output
    cv::exp(-masks_output, masks_output);
    cv::add(1.0, masks_output, masks_output);
    cv::divide(1.0, masks_output, masks_output);

    masks_output = masks_output.reshape((int)nms_result_size, {mask_height, mask_width});

    std::vector<cv::Mat> images((int)nms_result_size);
    cv::split(masks_output, images);

    for (size_t i = 0; i < detections.size(); i++) {
      Detection & detection = detections[i];
      int x1 = detection.box.x;
      int y1 = detection.box.y;
      int x2 = detection.box.x + detection.box.width;
      int y2 = detection.box.y + detection.box.height;

      int scale_x1 = x1 / x_factor_segm;
      int scale_y1 = y1 / y_factor_segm;
      int scale_x2 = x2 / x_factor_segm;
      int scale_y2 = y2 / y_factor_segm;

      cv::Mat crop_mask = images[i].colRange(scale_x1, scale_x2).rowRange(scale_y1, scale_y2);

      cv::resize(crop_mask, crop_mask, cv::Size(x2 - x1, y2 - y1));

      cv::Size kernel_size = cv::Size(
        int(x_factor_segm) % 2 ? int(x_factor_segm) : int(x_factor_segm) + 1,
        int(y_factor_segm) % 2 ? int(y_factor_segm) : int(y_factor_segm) + 1);
      cv::GaussianBlur(crop_mask, crop_mask, kernel_size, 0);
      cv::threshold(crop_mask, crop_mask, 0.5, 1, cv::THRESH_BINARY);

      // Convert crop_mask to 8UC1
      crop_mask.convertTo(crop_mask, CV_8UC1, 255);

      detection.mask = crop_mask;
    }
  }
  return detections;
}

/**
 * @brief Load ONNX network from file.
 */
void Inference::load_onnx_network()
{
  net_ = cv::dnn::readNetFromONNX(model_path_);

  if (cuda_enabled_) {
    if (verbose_) {
      std::cout << "\nRunning on CUDA" << std::endl;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  } else {
    if (verbose_) {
      std::cout << "\nRunning on CPU" << std::endl;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
}

} // namespace object_detector
