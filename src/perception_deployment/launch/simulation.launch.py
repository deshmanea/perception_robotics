import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from launch.actions import SetEnvironmentVariable



def generate_launch_description():
    pkg_dir = get_package_share_directory('perception_deployment')
    models_path = os.path.join(pkg_dir, 'models')
    
    world_file = os.path.join(models_path, 'warehouse_world.sdf')
    camera_model = os.path.join(models_path, 'realsense_rig/camera.sdf')

    return LaunchDescription([
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=[os.environ.get('GZ_SIM_RESOURCE_PATH', ''), ':', models_path]
        ),

        # 1. Start Gazebo Sim (New Command)
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_file],
            output='screen'
        ),

        # 2. Spawn the camera (New GZ Service call, replacing gazebo_ros)
        ExecuteProcess(
            cmd=[
                'gz', 'service', '-s', '/model/warehouse_world/create',
                '--reqtype', 'gz.msgs.EntityFactory',
                '--replytype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'file: "{camera_model}", name: "realsense"'
            ],
            output='screen'
        ),

        # 3. Modern Bridge (No changes needed here, your mapping is good)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '/camera/depth/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked'
            ],
            remappings=[
                ('image_raw', '/camera/image_raw'),
                ('depth_raw', '/camera/depth/image_raw'),
                ('camera_info', '/camera/camera_info'),
                ('target_3d', '/camera/depth/points')
            ],
            output='screen'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments = ['0', '0', '2', '0', '0', '0', 'world', 'my_camera/link/realsense_link'],
            parameters=[{'use_sim_time': True}]
        )
    ])