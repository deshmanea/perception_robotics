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
        self.fx = self.fy = self.cx = self.cy = None


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

            self.img_sub = message_filters.Subscriber(
                self, Image, '/realsense/image', qos_profile=qos_profile_sensor_data
            )

            self.depth_sub = message_filters.Subscriber(
                self, Image, '/realsense/depth_image', qos_profile=qos_profile_sensor_data
            )

            self.info_sub = self.create_subscription(
                CameraInfo,
                '/realsense/camera_info',
                self.info_cb,
                qos_profile_sensor_data
            )

            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.img_sub, self.depth_sub],
                queue_size=10,
                slop=0.05
            )

            self.ts.registerCallback(self.process_callback)

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
            if self.target_pub is None:
                self.get_logger().error("Publisher is None!")
                return TransitionCallbackReturn.FAILURE

            self.target_pub.on_activate(state)
            result = super().on_activate(state) 

            self.get_logger().info("Activation SUCCESS")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"Activation failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def info_cb(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def process_callback(self, img_msg, depth_msg):
        # The main 'Thin Spine' execution loop
        self.get_logger().info("📸 Image callback triggered")
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        # Run Inference
        conf = self.get_parameter('conf_threshold').value
        try:
            detections = self.trt.run(cv_img, conf)
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return
        
        if self.fx is None:
            return

        for det in detections:
            u = int((det[0] + det[2]) / 2)
            v = int((det[1] + det[3]) / 2)

            h, w = cv_depth.shape
            u = np.clip(u, 2, w - 3)
            v = np.clip(v, 2, h - 3)

            z = float(np.nanmedian(cv_depth[v-2:v+3, u-2:u+3]))
            if not np.isfinite(z) or z <= 0:
                continue
            
            if 0.2 < z < 5.0:
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                self.target_pub.publish(Point(x=x, y=y, z=z))

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating: Stopping data flow...")
        # Clean up subs to save CPU
        self.target_pub.on_deactivate()
        self.img_sub = self.depth_sub = self.ts = None
        return super().on_deactivate(state)
    
    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up: Destroying publisher...")
        self.destroy_publisher(self.target_pub)
        self.trt = None
        self.img_sub = None
        self.depth_sub = None
        self.ts = None
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