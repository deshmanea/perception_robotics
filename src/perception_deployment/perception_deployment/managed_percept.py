import rclpy
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ManagedPerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_engine')
        self.bridge = CvBridge()
        self.subscription = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring: Loading TensorRT Engine...')
        # Logic: This is where you'd load your .engine file to GPU memory
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating: Starting Inference...')
        # Only subscribe when active to save bandwidth/CPU
        self.subscription = self.create_subscription(
            Image,
            '/camera',
            self.image_callback,
            10)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating: Stopping Inference...')
        self.destroy_subscription(self.subscription)
        return TransitionCallbackReturn.SUCCESS

    def image_callback(self, msg):
        # This will be our future home for TensorRT inference
        self.get_logger().info('Processing frame at senior-level efficiency...', throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = ManagedPerceptionNode()
    rclpy.spin(node)
    rclpy.shutdown()