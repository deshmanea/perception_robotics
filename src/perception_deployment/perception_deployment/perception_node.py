#!/home/abhijit/.pyenv/versions/perception_env/bin/python
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
import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from perception_deployment.trt_bridge import TensorRTInference
import message_filters

class SpatialPerceptionNode(LifecycleNode):

    def __init__(self):
        super().__init__('perception_engine')
        self.bridge = CvBridge()
        self.get_logger().info("Node initialized in Unconfigured state")
        
        self.declare_parameter('engine_path', 'models/yolo.engine')
        self.declare_parameter('conf_threshold', 0.5)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring: Loading TensorRT Engine...")
        try:
            # 1. Always get the package directory first
            package_share = get_package_share_directory('perception_deployment')
            param_path = self.get_parameter('engine_path').value
            
            self.get_logger().info(f"Yolo Engine path > {param_path}")

            # 2. Construct the absolute path safely
            if param_path == '':
                engine_path = os.path.join(package_share, 'models', 'yolo.engine')
            else:
                # Now package_share is guaranteed to exist
                engine_path = os.path.join(package_share, param_path)

            self.get_logger().info(f"Loading TensorRT engine from: {engine_path}")
            
            # 3. Check and Load
            if not os.path.exists(engine_path):
                self.get_logger().error(f"Engine file NOT found at {engine_path}!")
                return TransitionCallbackReturn.FAILURE
                
            self.trt = TensorRTInference(engine_path)
            
            # Don't forget the publisher we discussed!
            self.target_pub = self.create_lifecycle_publisher(Point, 'perception/target_3d', 10)
            
            self.get_logger().info("Engine Loaded Successfully")    
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"Configuration failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating: Subscribing to Camera...")
        
        # 3. Synchronized Subscriptions
        self.img_sub = message_filters.Subscriber(self, Image, 'image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, 'depth_raw')
        self.info_sub = self.create_subscription(CameraInfo, 'camera_info', self.info_cb, 10)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.process_callback)

        return super().on_activate(state)

    def info_cb(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def process_callback(self, img_msg, depth_msg):
        # The main 'Thin Spine' execution loop
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        # Run Inference
        conf = self.get_parameter('conf_threshold').value
        detections = self.trt.run(cv_img, conf)

        for det in detections:
            # Spatial Math (2D -> 3D)
            u, v = int((det[0]+det[2])/2), int((det[1]+det[3])/2)
            z = float(np.nanmedian(cv_depth[v-2:v+3, u-2:u+3]))
            
            if 0.2 < z < 5.0:
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                self.target_pub.publish(Point(x=x, y=y, z=z))

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating: Stopping data flow...")
        # Clean up subs to save CPU
        self.img_sub = self.depth_sub = self.ts = None
        return super().on_deactivate(state)
    
    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up: Destroying publisher...")
        self.destroy_publisher(self.target_pub)
        return TransitionCallbackReturn.SUCCESS


def main(args=None):
    rclpy.init(args=args)
    node = SpatialPerceptionNode() # Ensure this matches your class name
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()