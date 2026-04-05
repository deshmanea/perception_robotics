from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'perception_deployment'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[
        'perception_deployment',
        'perception_deployment.*'
    ]),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/simulation.launch.py']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
        (os.path.join('share', package_name, 'models'), ['models/yolo.engine']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models/realsense_rig'), glob('models/realsense_rig/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abhijit',
    description='Perception Thin Spine: Managed TensorRT Perception',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = perception_deployment.perception_node:main',
        ],
    },
)