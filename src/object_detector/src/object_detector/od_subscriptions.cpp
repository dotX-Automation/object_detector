/**
 * Object Detector node topic subscription callbacks.
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
 * @brief Parses a new synchronized image message.
 *
 * @param image_msg Image message to parse.
 * @param depth_msg Depth image message to parse.
 */
void ObjectDetectorNode::sync_callback(const Image::ConstSharedPtr & image_msg,
                                       const Image::ConstSharedPtr & depth_msg)
{
  UNUSED(depth_msg);

  // Convert msg to OpenCV image
  cv::Mat frame = cv::Mat(
    image_msg->height,
    image_msg->width,
    CV_8UC3,
    (void *)(image_msg->data.data()));

  // Pass data to worker thread
  sem_wait(&sem1_);
  new_frame_ = frame.clone();
  last_header_ = image_msg->header;
  sem_post(&sem2_);
}

} // namespace ObjectDetector
