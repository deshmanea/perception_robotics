#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import os
from threading import Lock
from ament_index_python.packages import get_package_share_directory
import cv2

# --- TensorRT pipeline imports ---
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from perception_deployment.trt_bridge import TensorRTInference

class SensorBuffer:
    def __init__(self):
        self.lock = Lock()
        self.image = None
        self.depth = None

    def update_image(self, img):
        with self.lock:
            self.image = img

    def update_depth(self, depth):
        with self.lock:
            self.depth = depth

    def get_pair(self):
        with self.lock:
            return self.image, self.depth


class SpatialPerceptionNode(LifecycleNode):

    def __init__(self):
        super().__init__('perception_engine')

        self.bridge = CvBridge()
        self.buffer = SensorBuffer()
        
        self.is_active = False

        self.fx = self.fy = self.cx = self.cy = None

        self.get_logger().info("Node initialized in Unconfigured state")
        
        self.declare_parameter('engine_path', 'models/yolo.engine')
        self.declare_parameter('conf_threshold', 0.01)
        
        self.trt = None
        self.timer = None

        self.get_logger().info("Node initialized (UNCONFIGURED)")


    def on_configure(self, state: State) -> TransitionCallbackReturn:
        
        self.get_logger().info("Configuring: Loading TensorRT Engine...")

        try:
            package_share = get_package_share_directory('perception_deployment')
            param_path = self.get_parameter('engine_path').value
            
            self.get_logger().info(f"Yolo Engine path > {param_path}")

            if param_path == '':
                engine_path = os.path.join(package_share, 'models', 'yolo.engine')
            elif os.path.isabs(param_path):
                engine_path = param_path
            else:
                engine_path = os.path.join(package_share, param_path)
            
            self.get_logger().info(f"Engine path: {engine_path}")

            if not os.path.exists(engine_path):
                self.get_logger().error(f"Engine not found: {engine_path}")
                return TransitionCallbackReturn.FAILURE

            self.trt = TensorRTInference(engine_path)

            self.target_pub = self.create_lifecycle_publisher(Point, '/perception/target_3d', 10)
            
            self.get_logger().info("Engine Loaded Successfully")   

            self.info_sub = self.create_subscription(
                CameraInfo,
                '/realsense/camera_info',
                self.info_cb,
                qos_profile_sensor_data
            )

            self.img_sub = self.create_subscription(
                Image,
                '/realsense/image',
                self.image_cb,
                qos_profile_sensor_data
            )

            self.depth_sub = self.create_subscription(
                Image,
                '/realsense/depth_image',
                self.depth_cb,
                qos_profile_sensor_data
            )

            self.timer = self.create_timer(0.03, self.process_loop)

            self.get_logger().info("Subscriptions ready")

            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            import traceback
            self.get_logger().error(f"Configuration failed: {str(e)}")
            self.get_logger().error(traceback.format_exc())
            self.get_logger().error(f"Configuration failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating node...")
         
        try:
            self.target_pub.on_activate(state)
            self.is_active = True

            self.get_logger().info("Activated successfully")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"Activation failed: {e}")
            self.is_active = False
            return TransitionCallbackReturn.FAILURE
    
    def image_cb(self, msg):
        if not self.is_active:
            return
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.buffer.update_image(img)

    def depth_cb(self, msg):
        if not self.is_active:
            return
        depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.buffer.update_depth(depth)

    def info_cb(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def process_loop(self):
        if not self.is_active:
            return

        img, depth = self.buffer.get_pair()

        if img is None or depth is None:
            self.get_logger().warn("No synced data yet")
            return

        try:
            conf_th = self.get_parameter('conf_threshold').value

            img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            detections = self.trt.run(img_input, 0.001)

            if detections is None or len(detections) == 0:
                return

            detections = np.array(detections, dtype=np.float32)
            if detections.ndim == 1:
                if detections.size % 6 != 0:
                    self.get_logger().error(f"Invalid detection size: {detections.size}")
                    return
                detections = detections.reshape(-1, 6)

            self.get_logger().info(f"Num detections: {len(detections)}")
            self.get_logger().info(f"Max conf: {np.max(detections[:, 4])}")

            detections = detections[detections[:, 4] >= conf_th]

            if len(detections) == 0:
                return

            if self.fx is None:
                self.get_logger().error("Camera intrinsics not set")
                return

            h, w = depth.shape[:2]

            for det in detections:

                x, y, bw, bh, obj_conf, cls = det

                x1 = x - bw / 2
                y1 = y - bh / 2
                x2 = x + bw / 2
                y2 = y + bh / 2

                # clamp bbox
                x1 = int(np.clip(x1, 0, w - 1))
                y1 = int(np.clip(y1, 0, h - 1))
                x2 = int(np.clip(x2, 0, w - 1))
                y2 = int(np.clip(y2, 0, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                roi = depth[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                z = float(np.nanmedian(roi))

                if not np.isfinite(z) or z <= 0:
                    continue

                if not (0.2 < z < 5.0):
                    continue

                u = (x1 + x2) // 2
                v = (y1 + y2) // 2

                X = (u - self.cx) * z / self.fx
                Y = (v - self.cy) * z / self.fy

                self.target_pub.publish(Point(x=X, y=Y, z=z))

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating...")

        self.is_active = False

        if self.timer:
            self.destroy_timer(self.timer)
            self.timer = None

        if hasattr(self, 'img_sub') and self.img_sub:
            self.destroy_subscription(self.img_sub)
            self.img_sub = None

        if hasattr(self, 'depth_sub') and self.depth_sub:
            self.destroy_subscription(self.depth_sub)
            self.depth_sub = None

        self.target_pub.on_deactivate()

        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up: Destroying publisher...")
        self.destroy_publisher(self.target_pub)
        self.trt = None
        self.img_sub = None
        self.depth_sub = None
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state):
        self.get_logger().info("Shutting down node")
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = SpatialPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()