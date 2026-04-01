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
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
        (os.path.join('share', package_name), ['perception_deployment/yolo.engine']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abhijit',
    maintainer_email='your@email.com',
    description='Senior Portfolio: Managed Perception Pipeline',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'managed_percept = perception_deployment.managed_percept:main',
            'perception_node = perception_deployment.perception_node:main',
        ],
    },
)