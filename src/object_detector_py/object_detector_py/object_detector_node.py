import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

from threading import Thread, Lock, Semaphore

from ultralytics import YOLO

from rclpy.node import Node

import dua_qos_py.dua_qos_besteffort as dua_qos_besteffort
import dua_qos_py.dua_qos_reliable as dua_qos_reliable

from message_filters import Subscriber, TimeSynchronizer

from std_msgs.msg import Header
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

class AtomicBool:
    def __init__(self, initial=False):
        self.value = initial
        self._lock = Lock()

    def store(self, new_value):
        with self._lock:
            self.value = new_value

    def load(self):
        with self._lock:
            return self.value

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')

        self.init_parameters()
        self.init_atomics()
        self.init_publishers()
        self.init_subscriptions()
        self.init_services()
        self.init_network()

        # Initialize other variables
        self.last_header = Header()
        self.bridge = CvBridge()

        self.target_classes_ids = [i for i in self.model.names if self.model.names[i] in self.classes_targets]

        self.get_logger().info('Node initialized')

    def cleanup(self):
        if self.running.load():
            # Join worker thread
            self.running.store(False)
            self.sem1.release()
            self.sem2.release()
            self.worker.join()

            # Shutdown subscriptions
            if self.use_depth:
                del self.sync.callbacks[0]
                self.destroy_subscription(self.sync)
                self.destroy_subscription(self.image_sub_sync)
                self.destroy_subscription(self.depth_sub_sync)
            else:
                self.image_sub.destroy()

        self.stream_pub.destroy()

    def init_parameters(self):
        """
        Init parameters
        """
        self.declare_parameters(
            namespace='',
            parameters=[('always_pub_stream', False),
                        ('autostart', False),
                        ('best_effort_sub_qos', False),
                        ('classes', ['']),
                        ('classes_targets', ['']),
                        ('image_sub_depth', 1),
                        ('model_score_threshold', 0.1),
                        ('model_shape', [100, 100]),
                        ('pt_path', ''),
                        ('use_depth', False)
                       ])

        self.always_pub_stream = self.get_parameter('always_pub_stream').value
        self.autostart = self.get_parameter('autostart').value
        self.best_effort_sub_qos = self.get_parameter('best_effort_sub_qos').value
        self.classes = self.get_parameter('classes').value
        self.classes_targets = self.get_parameter('classes_targets').value
        self.image_sub_depth = self.get_parameter('image_sub_depth').value
        self.model_score_threshold = self.get_parameter('model_score_threshold').value
        self.model_shape = self.get_parameter('model_shape').value
        self.pt_path = self.get_parameter('pt_path').value
        self.use_depth = self.get_parameter('use_depth').value

        self.get_logger().info(f'always_pub_stream: {self.always_pub_stream}')
        self.get_logger().info(f'autostart: {self.autostart}')
        self.get_logger().info(f'best_effort_sub_qos: {self.best_effort_sub_qos}')
        self.get_logger().info(f'classes: {self.classes}')
        self.get_logger().info(f'classes_targets: {self.classes_targets}')
        self.get_logger().info(f'image_sub_depth: {self.image_sub_depth}')
        self.get_logger().info(f'model_score_threshold: {self.model_score_threshold}')
        self.get_logger().info(f'model_shape: {self.model_shape}')
        self.get_logger().info(f'pt_path: {self.pt_path}')
        self.get_logger().info(f'use_depth: {self.use_depth}')

    def init_atomics(self):
        """
        Init atomics
        """
        self.running = AtomicBool(initial=False)

    def init_publishers(self):
        """
        Init publishers
        """
        # Detections
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/detections',
            dua_qos_reliable.get_datum_qos())

        # Detections stream
        self.stream_pub = self.create_publisher(
            Image,
            '/detections_stream',
            dua_qos_besteffort.get_image_qos())

    def init_subscriptions(self):
        """
        Init subscriptions
        """
        if self.autostart:
            # Initialize semaphores
            self.sem1 = Semaphore(1)
            self.sem2 = Semaphore(0)

            # Spawn worker thread
            self.running.store(True)
            self.worker = Thread(target=self.worker_thread_routine)
            self.worker.start()

            qos_profile = dua_qos_besteffort.get_image_qos(depth=self.image_sub_depth) if self.best_effort_sub_qos else dua_qos_reliable.get_image_qos(depth=self.image_sub_depth)
            if self.use_depth:
                self.image_sub_sync = Subscriber(
                    self,
                    Image,
                    '/image',
                    qos_profile=qos_profile)

                self.depth_sub_sync = Subscriber(
                    self,
                    Image,
                    '/depth_distances',
                    qos_profile=qos_profile)

                # Creazione di un sincronizzatore ExactTime
                self.sync = TimeSynchronizer([self.image_sub_sync, self.depth_sub_sync], queue_size=1)
                self.sync.registerCallback(self.sync_callback)
            else:
                self.image_sub = self.create_subscription(
                    Image,
                    '/image',
                    self.image_callback,
                    qos_profile=qos_profile)

    def init_services(self):
        # Enable
        self.enable_server = self.create_service(SetBool, '~/enable', self.enable_callback)

    def init_network(self):
        self.model = YOLO(self.pt_path, verbose=False)

    def worker_thread_routine(self):
        """
        Worker thread routine
        """
        while True:
            depth = None
            self.sem2.acquire()
            if not self.running.load():
                break
            header = self.last_header
            image = self.new_frame.copy()
            if self.use_depth:
                depth = self.new_depth.copy()
            self.sem1.release()

            # Detect targets
            nn_results = self.model(image,
                                 imgsz=self.model_shape,
                                 stream=True,
                                 verbose=False,
                                 conf=self.model_score_threshold,
                                 classes=self.target_classes_ids)

            for nn_result in nn_results:
                detections = len(nn_result)

                if detections == 0 and not self.always_pub_stream:
                    continue

                if detections > 0:
                    detections_msg = Detection2DArray()
                    detections_msg.header = header

                    for i in range(detections):
                        box = nn_result.boxes[i]
                        conf = box.conf[0].cpu().item()
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        label = f'{self.model.names[cls]} {conf:.2f}'

                        # Draw bbox and label
                        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if nn_result.masks is not None:
                            mask = nn_result.masks[i].data.cpu().squeeze().numpy().astype(np.uint8)
                            mask_resized = cv.resize(mask, (image.shape[1], image.shape[0]))
                            mask_rgb = cv.cvtColor(mask_resized*255, cv.COLOR_GRAY2RGB)
                            cv.addWeighted(image, 1.0, mask_rgb, 0.3, 0, image)
                            mask_roi = mask_resized[y1:y2, x1:x2]

                        # Detection message
                        detection_msg = Detection2D()
                        # Set htpothesis
                        result = ObjectHypothesisWithPose()
                        result.hypothesis.class_id = self.model.names[cls]
                        result.hypothesis.score = conf

                        if depth is not None:
                            result.pose.covariance[0] = 1.0

                            depth_roi = depth[y1:y2, x1:x2]
                            if nn_result.masks is not None:
                                depth_roi = depth_roi * mask_roi
                                valid_pixels = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
                            else:
                                valid_pixels = depth_roi[np.isfinite(depth_roi)]

                            sum_valid = np.sum(valid_pixels)
                            count_valid = valid_pixels.size

                            if count_valid == 0:
                                continue
                            mean = sum_valid / count_valid

                            # Compute centroid of the bounding box expressed wrt frame center
                            u = (x1 + x2 - image.shape[1]) / 2 / image.shape[1]
                            v = (y1 + y2 - image.shape[0]) / 2 / image.shape[0]

                            Z = mean / np.sqrt(u**2 + v**2 + 1)
                            X = Z * u
                            Y = Z * v

                            result.pose.pose.position.x = X
                            result.pose.pose.position.y = Y
                            result.pose.pose.position.z = Z

                            print(mean)
                            print('u: ', u, 'v: ', v)
                            print('x: ', X, 'y: ', Y, 'z: ', Z)
                        else:
                            result.pose.covariance[0] = -1.0

                        detection_msg.results.append(result)

                        # Set bounding box
                        detection_msg.header = header
                        detection_msg.bbox.center.position.x = (x1 + x2) / 2
                        detection_msg.bbox.center.position.y = (y1 + y2) / 2
                        detection_msg.bbox.size_x = float(x2 - x1)
                        detection_msg.bbox.size_y = float(y2 - y1)

                        detections_msg.detections.append(detection_msg)

                    self.detections_pub.publish(detections_msg)

                camera_frame = image
                processed_image_msg = self.bridge.cv2_to_imgmsg(camera_frame, encoding="bgr8")
                processed_image_msg.header = header

                self.stream_pub.publish(processed_image_msg)

    def image_callback(self, image_msg):
        """
        Image callback
        """
        print('image_callback')
        # Convert image_msg to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(image_msg)

        # Pass data to worker thread
        self.sem1.acquire()
        self.new_frame = frame.copy()
        self.last_header = image_msg.header
        self.sem2.release()

    def sync_callback(self, image_msg, depth_msg):
        """
        Sync callback
        """
        print('sync_callback')
        # Convert image_msg to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(image_msg)

        # Convert depth_msg to OpenCV image
        depth = self.bridge.imgmsg_to_cv2(depth_msg)

        # Pass data to worker thread
        self.sem1.acquire()
        self.new_frame = frame.copy()
        self.new_depth = depth.copy()
        self.last_header = image_msg.header
        self.sem2.release()

    def enable_callback(self, req, resp):
        """
        Enable callback
        """
        if req.data:
            if not self.running.load():
                # Initialize semaphores
                self.sem1 = Semaphore(1)
                self.sem2 = Semaphore(0)

                # Spawn worker thread
                self.running.store(True)
                self.worker = Thread(target=self.worker_thread_routine)
                self.worker.start()

                qos_profile = dua_qos_besteffort.get_image_qos(depth=self.image_sub_depth) if self.best_effort_sub_qos else dua_qos_reliable.get_image_qos(depth=self.image_sub_depth)

                if self.use_depth:
                    self.image_sub_sync = Subscriber(
                        self,
                        Image,
                        '/image',
                        qos_profile=qos_profile)

                    self.depth_sub_sync = Subscriber(
                        self,
                        Image,
                        '/depth_distances',
                        qos_profile=qos_profile)

                    # Initialize synchronizer
                    self.sync = TimeSynchronizer([self.image_sub_sync, self.depth_sub_sync], queue_size=1)
                    self.sync.registerCallback(self.sync_callback)
                else:
                    self.image_sub = self.create_subscription(
                        Image,
                        '/image',
                        self.image_callback,
                        qos_profile=qos_profile)

                self.get_logger().info('Object detector ACTIVATED')

            resp.success = True
            resp.message = ''
        else:
            if self.running.load():
                # Join worker thread
                self.running.store(False)
                self.sem1.release()
                self.sem2.release()
                self.worker.join()

                # Shutdown subscriptions
                if self.use_depth:
                    del self.sync.callbacks[0]
                    self.destroy_subscription(self.sync)
                    self.destroy_subscription(self.image_sub_sync)
                    self.destroy_subscription(self.depth_sub_sync)
                else:
                    self.destroy_subscription(self.image_sub)

                del self.sem1
                del self.sem2

            resp.success = True
            resp.message = ''

            self.get_logger().info('Object detector DEACTIVATED')

        return resp

