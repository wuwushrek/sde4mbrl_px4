from setuptools import setup

package_name = 'mpc4px4'

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
    author='Franck Djeumou',
    author_email='frmbouwe@gmail.com',
    maintainer='Franck Djeumou',
    maintainer_email='frmbouwe@gmail.com',
    description='A package for quadrotor modeling and control using Model Predictive Control',
    license='GNU Public License',
    tests_require=['pytest', 'numpy', 'scipy', 'matplotlib'],
    entry_points={
        'console_scripts': [
        ],
    },
)
