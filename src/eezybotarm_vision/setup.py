from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'eezybotarm_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.jpg')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='vitalii',
    maintainer_email='mostipaka.vitalii@gmail.com',
    description='Computer vision and sorting coordination package for EezyBotArm',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sorting_node = eezybotarm_vision.sorting_node:main',
        ],
    },
)
