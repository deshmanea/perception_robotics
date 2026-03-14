import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import qos_profile_sensor_data # Added for Jazzy/Gazebo compatibility
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import CameraInfo

class SpatialPerceptionNode(Node):

    def __init__(self):
            super().__init__('spatial_engine')
            self.bridge = CvBridge()
            self.latest_depth = None
            self.fx = None
            self.fy = None
            self.cx = None
            self.cy = None

            # Individual subscribers (No more filter)
            self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data)
            self.depth_sub = self.create_subscription(Image, '/camera/depth', self.depth_callback, qos_profile_sensor_data)

            self.target_pub = self.create_publisher(Point, '/perception/target_3d', 10)
            self.model = YOLO("yolo26n.pt") 
            
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                '/camera/camera_info',
                self.camera_info_callback,
                qos_profile_sensor_data
            )

            self.get_logger().info("--- SPINE Block 3: Asynchronous Mode ---")

           

    def depth_callback(self, msg):
        # Just store the most recent depth map
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def image_callback(self, msg):
        if self.latest_depth is None:
            self.get_logger().warn("Waiting for camera intrinsics...")
            return

        cv_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_rgb, verbose=False, device='cuda')

        for result in results:
            if len(result.boxes) == 0:
                return

            # Take first detection
            box = result.boxes[0]
            
            # --- Confidence Filtering ---
            conf = float(box.conf[0])
            if conf < 0.25:
                self.get_logger().info(f"Low confidence ({conf:.2f}) — skipping")
                return

            xywh = box.xywh[0].cpu().numpy()
            u, v = int(xywh[0]), int(xywh[1])

            # --- Bounds Check ---
            h, w = self.latest_depth.shape
            if not (2 <= v < h-2 and 2 <= u < w-2):
                self.get_logger().warn("Detection too close to edge — skipping")
                return

            # --- 5x5 Median Depth Window ---
            window = self.latest_depth[v-2:v+3, u-2:u+3]
            depth = np.nanmedian(window)

            if np.isnan(depth) or depth <= 0.1:
                self.get_logger().warn("Invalid depth — skipping")
                return

            # --- Pinhole Projection ---
            z = float(depth)
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy

            p = Point(x=x, y=y, z=z)
            self.target_pub.publish(p)

            self.get_logger().info(
                f"📍 Target | Conf:{conf:.2f} | X:{x:.2f} Y:{y:.2f} Z:{z:.2f}"
            )
                

    def gt_callback(self, msg):
        for pose in msg.poses:
            if pose.name == "vis_cylinder":
                self.gt_position = np.array([
                    pose.position.x,
                    pose.position.y,
                    pose.position.z
                ])
    
    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]


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