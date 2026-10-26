from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'eezybotarm_debug'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vitalii',
    maintainer_email='mostipaka.vitalii@gmail.com',
    description='EezyBotArm development and debugging visualization tools',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'workspace_visualizer = eezybotarm_debug.workspace_visualizer:main',
        ],
    },
)
