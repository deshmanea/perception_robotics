import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import qos_profile_sensor_data # Added for Jazzy/Gazebo compatibility
import numpy as np
from ultralytics import YOLO

class SpatialPerceptionNode(Node):

    def __init__(self):
            super().__init__('spatial_engine')
            self.bridge = CvBridge()
            self.latest_depth = None

            # Individual subscribers (No more filter)
            self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data)
            self.depth_sub = self.create_subscription(Image, '/camera/depth', self.depth_callback, qos_profile_sensor_data)

            self.target_pub = self.create_publisher(Point, '/perception/target_3d', 10)
            self.model = YOLO("yolo26n.pt") 
            
            self.fx, self.fy = 615.0, 615.0
            self.cx, self.cy = 320.0, 240.0
            self.get_logger().info("--- SPINE Block 3: Asynchronous Mode ---")

    def depth_callback(self, msg):
        # Just store the most recent depth map
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def image_callback(self, msg):
        if self.latest_depth is None:
            self.get_logger().warn("Waiting for first depth map...")
            return

        cv_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_rgb, verbose=False, device='cuda')

        for result in results:
            num_boxes = len(result.boxes)
            
            if num_boxes == 0:
                 # This tells us the script is working, but YOLO is "blind" right now
                 self.get_logger().info("Scanning... (No objects detected in frame)", once=True)

            if len(result.boxes) > 0:
                box = result.boxes[0].xywh[0].cpu().numpy()
                u, v = int(box[0]), int(box[1])

                # Use the latest depth map we have
                depth = self.latest_depth[v, u]
                
                if not np.isnan(depth) and depth > 0.1:
                    z = float(depth)
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    
                    p = Point(x=x, y=y, z=z)
                    self.target_pub.publish(p)
                    self.get_logger().info(f"📍 Target at: X:{x:.2f}m, Y:{y:.2f}m, Z:{z:.2f}m")


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