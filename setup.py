import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


class PostInstallCommand(install):
    """Post-installation command."""
    def run(self):
        subprocess.call(["pip3", "uninstall", "nvidia_cublas_cu11"])
        install.run(self)

setup(
    name='cnn-nas',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch==1.13.1',
        'torchvision==0.14.1',
        'nni==2.7',
        'pytorch-lightning==1.9.5',
        'filelock==3.10'
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    entry_points={
        'console_scripts': [
            'main = main:__main__'
        ]
    },
    author='Jan Olivetti and Carla Morral',
    author_email=['jo2708@columbia.edu', 'cm4257@columbia.edu'],
    description='An Exploration of Efficient CNN Design using Neural Architecture Search using Microsoft\'s Neural Network Intelligence library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/carlaMorral/CNN-NAS',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
