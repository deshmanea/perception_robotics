
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode

def generate_launch_description():
    pkg_share = get_package_share_directory('perception_deployment')
    config_file = os.path.join(pkg_share, 'config', 'params.yaml')

    # Define the Managed Node
    perception_node = LifecycleNode(
        package='perception_deployment',
        executable='perception_node',
        name='perception_engine',
        namespace='',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('image_raw', '/camera/image_raw'),
            ('depth_raw', '/camera/depth/image_raw'),
            ('camera_info', '/camera/camera_info'),
            ('target_3d', '/perception/target_3d')
        ]
    )

    return LaunchDescription([perception_node])