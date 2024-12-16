from setuptools import setup

package_name = 'yolo_pointcloud_marker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='YOLO object detection visualization in 3D using point cloud and markers.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_pointcloud_marker_node = yolo_pointcloud_marker.yolo_pointcloud_marker_node:main'
        ],
    },
)
