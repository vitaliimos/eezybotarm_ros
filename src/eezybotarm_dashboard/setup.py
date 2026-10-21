import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'eezybotarm_dashboard'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'nicegui',
        'numpy',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='Vitalii Mostipaka',
    maintainer_email='mostipaka.vitalii@gmail.com',
    description='Web dashboard for EezyBotArm using NiceGUI',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dashboard_node = eezybotarm_dashboard.dashboard_node:main',
        ],
    },
)
