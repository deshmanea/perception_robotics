import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    # Path to your SDF model
    model_path = os.path.expanduser('~/robotics_ws/src/perception_deployment/models/camera.sdf')

    return LaunchDescription([
        # 1. Start Gazebo with a warehouse (high complexity for perception)
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', 'warehouse.sdf'],
            output='screen'
        ),

        # 2. Spawn the camera rig
        ExecuteProcess(
            cmd=['gz', 'service', '-s', '/world/warehouse/create',
                 '--reqtype', 'gz.msgs.EntityFactory',
                 '--replytype', 'gz.msgs.Boolean',
                 '--timeout', '1000',
                 '--data', f'file: "{model_path}", name: "realsense"'],
            output='screen'
        ),

        # 3. Bridge (Maps ALL sensor streams)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/realsense/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/realsense/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/realsense/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                '/realsense/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
            ],
            output='screen'
        )
    ])