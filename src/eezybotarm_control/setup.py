from setuptools import find_packages, setup

package_name = 'eezybotarm_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vitalii',
    maintainer_email='mostipaka.vitalii@gmail.com',
    description='MoveIt 2 motion planning and pick-and-place action server for EezyBotArm',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pick_and_place_server = eezybotarm_control.pick_and_place_server:main',
        ],
    },
)
