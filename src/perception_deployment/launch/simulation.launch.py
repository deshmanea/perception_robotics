import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    # Paths
    warehouse_world = os.path.expanduser(
        '~/robotics_ws/src/perception_deployment/models/warehouse.sdf'
    )
    camera_model = os.path.expanduser(
        '~/robotics_ws/src/perception_deployment/models/camera.sdf'
    )

    # Fallback: if warehouse.sdf doesn't exist, use empty.world
    if not os.path.exists(warehouse_world):
        print("[WARNING] warehouse.sdf not found. Using empty world.")
        warehouse_world = '/opt/ros/jazzy/share/gazebo_ros/worlds/empty.world'

    return LaunchDescription([
        # 1. Start Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', '-v', '4', warehouse_world],
            output='screen'
        ),

        # 2. Spawn the camera using ROS2 spawn_node (reliable)
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-file', camera_model,
                        '-entity', 'realsense'
                    ],
                    output='screen'
                )
            ]
        )

        # 3. Bridge sensor topics
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/realsense/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/realsense/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/realsense/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
            ],
            remappings=[
                ('/realsense/image', '/camera/image_raw'),
                ('/realsense/depth_image', '/camera/depth/image_raw'),
                ('/realsense/camera_info', '/camera/camera_info')
            ],
            output='screen'
        )
    ])