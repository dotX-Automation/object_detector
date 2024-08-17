/**
 * Object Detector node service callbacks.
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

#include <object_detector/object_detector.hpp>

namespace ObjectDetector
{

/**
 * @brief Toggles target detection.
 *
 * @param req Service request to parse.
 * @param rest Service response to populate.
 */
void ObjectDetectorNode::enable_callback(
  SetBool::Request::SharedPtr req,
  SetBool::Response::SharedPtr resp)
{
  if (req->data) {
    if (!running_.load(std::memory_order_acquire)) {
      // Initialize semaphores
      sem_init(&sem1_, 0, 1);
      sem_init(&sem2_, 0, 0);

      // Spawn camera thread
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
          char * err_msg = strerror_r(errno, err_msg_buf, 100);
          throw std::runtime_error(
            "ObjectDetectorNode::enable_callback: Failed to configure worker thread: " +
            std::string(err_msg));
        }
      }

      if (use_depth_)
      {
        image_sub_sync_ = std::make_shared<image_transport::SubscriberFilter>();
        depth_sub_sync_ = std::make_shared<image_transport::SubscriberFilter>();

        // Subscribe to image topic
        image_sub_sync_->subscribe(
          this,
          "/image",
          transport_,
          best_effort_sub_qos_ ?
          dua_qos::BestEffort::get_image_qos(image_sub_depth_).get_rmw_qos_profile() :
          dua_qos::Reliable::get_image_qos(image_sub_depth_).get_rmw_qos_profile());

        // Subscribe to depth topic
        depth_sub_sync_->subscribe(
          this,
          "/depth_distances",
          transport_,
          best_effort_sub_qos_ ?
          dua_qos::BestEffort::get_image_qos(image_sub_depth_).get_rmw_qos_profile() :
          dua_qos::Reliable::get_image_qos(image_sub_depth_).get_rmw_qos_profile());

        // Initialize synchronizer
        sync_ = std::make_shared<message_filters::Synchronizer<depth_sync_policy>>(
          depth_sync_policy(10),
          *image_sub_sync_,
          *depth_sub_sync_);
        sync_->registerCallback(&ObjectDetectorNode::sync_callback, this);
      }
      else
      {
        image_sub_ = std::make_shared<image_transport::Subscriber>(
          image_transport::create_subscription(
            this,
            "/image",
            std::bind(
              &ObjectDetectorNode::image_callback,
              this,
              std::placeholders::_1),
            transport_,
            best_effort_sub_qos_ ?
            dua_qos::BestEffort::get_image_qos(image_sub_depth_).get_rmw_qos_profile() :
            dua_qos::Reliable::get_image_qos(image_sub_depth_).get_rmw_qos_profile()));
      }

      RCLCPP_WARN(this->get_logger(), "Object Detector ACTIVATED");
    }
    resp->set__success(true);
    resp->set__message("");
  } else {
    if (running_.load(std::memory_order_acquire)) {
      // Join worker thread
      running_.store(false, std::memory_order_release);
      sem_post(&sem1_);
      sem_post(&sem2_);
      worker_.join();

      // Shutdown subscriptions
      if (use_depth_)
      {
        sync_.reset();
        image_sub_sync_.reset();
        depth_sub_sync_.reset();
      }
      else
      {
        image_sub_.reset();
      }

      // Destroy semaphores
      sem_destroy(&sem1_);
      sem_destroy(&sem2_);
    }
    resp->set__success(true);
    resp->set__message("");
  }
}

} // namespace ObjectDetector
