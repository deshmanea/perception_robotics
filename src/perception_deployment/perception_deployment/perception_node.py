import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

# --- TensorRT pipeline imports ---
from perception_deployment.trt.engine import load_engine
from perception_deployment.trt.preprocess import preprocess
from perception_deployment.trt.infer import allocate_buffers, infer
from perception_deployment.trt.postprocess import filter_detections, scale_boxes


class SpatialPerceptionNode(Node):

    def __init__(self):
        super().__init__('spatial_engine')

        # --- Core utilities ---
        self.bridge = CvBridge()

        # --- Sensor state ---
        self.latest_depth = None
        self.fx = self.fy = self.cx = self.cy = None

        # --- Subscribers ---
        self.image_sub = self.create_subscription(
            Image, '/realsense/image', self.image_callback, qos_profile_sensor_data
        )

        self.depth_sub = self.create_subscription(
            Image, '/realsense/depth_image', self.depth_callback, qos_profile_sensor_data
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/realsense/camera_info', self.camera_info_callback, qos_profile_sensor_data
        )

        # --- Publisher ---
        self.target_pub = self.create_publisher(Point, '/perception/target_3d', 10)

        # --- TensorRT initialization (loaded once) ---
        pkg_path = get_package_share_directory('perception_deployment')
        engine_path = os.path.join(pkg_path, 'yolo.engine')
        self.engine, self.trt_context, self.trt_runtime = load_engine(engine_path)
        self.inputs, self.outputs, self.bindings = allocate_buffers(self.engine)

        self.get_logger().info("Spatial Perception Node (TensorRT) Ready")

    # -------------------------
    # Depth Callback
    # -------------------------
    def depth_callback(self, msg):
        # Store latest depth frame (meters, float32)
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    # -------------------------
    # Image Callback (Main Pipeline)
    # -------------------------
    def image_callback(self, msg):
        self.get_logger().info("Image callback triggered")

        # --- Ensure required data available ---
        if self.latest_depth is None or self.fx is None:
            return

        # --- Convert ROS Image → OpenCV ---
        cv_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # --- Preprocess (resize + normalize + letterbox assumed) ---
        self.get_logger().info("Image converted")
        img, orig = preprocess(cv_bgr)
        self.get_logger().info("Preprocess done")

        # --- TensorRT Inference ---
        output = infer(self.trt_context, self.bindings, self.inputs, self.outputs, img)
        self.get_logger().info("Inference done")

        # --- Postprocess (NMS + scaling back to original image) ---
        filtered = filter_detections(output)
        self.get_logger().info(f"Filter done: {len(filtered)}")

        detections = scale_boxes(filtered, orig.shape)

        self.get_logger().info("Scale done")
        if len(detections) == 0:
            return

        # --- Use first detection ---
        x1, y1, x2, y2, conf, cls = detections[0]

        if conf < 0.25:
            return

        # --- Compute center pixel ---
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        # --- Bounds check ---
        h, w = self.latest_depth.shape
        if not (2 <= v < h - 2 and 2 <= u < w - 2):
            return

        # --- Depth extraction (5x5 median filter) ---
        window = self.latest_depth[v - 2:v + 3, u - 2:u + 3]
        depth = np.nanmedian(window)

        if np.isnan(depth) or depth <= 0.1:
            return

        # --- Pinhole projection (2D → 3D) ---
        z = float(depth)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # --- Publish 3D point ---
        point = Point(x=x, y=y, z=z)
        self.target_pub.publish(point)

        self.get_logger().info(
            f"Target | Conf:{conf:.2f} | X:{x:.2f} Y:{y:.2f} Z:{z:.2f}"
        )

    # -------------------------
    # Camera Intrinsics Callback
    # -------------------------
    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]


# -------------------------
# Main Entry
# -------------------------
def main():
    rclpy.init()
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