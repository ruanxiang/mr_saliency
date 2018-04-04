"""setup for mr_saliency to enable pip install"""
import os, sys
from setuptools import setup, find_packages

def readme():
    """opens github readme"""
    with open('README.md') as _fo:
        return _fo.read()

def readversion():
    """versioning of pip installer"""
    with open('version.py') as _fo:
        return _fo.read().split(' = ')[1]

def setup_package():
    """ setup"""

    metadata = dict(
        name='MR',
        version=readversion(),
        description='implementation of mr_saliency',
        url='https://github.com/xkunglu/mr_saliency',
        author='xkunglu -forked from ruanxiang',
        author_email='xkunglu@gmail.com',
        license='GPL2',
        dependency_links=['cv2', 'numpy', 'skimage', 'matplotlib', 'scipy', 'wxPython'],
        packages=find_packages(),
        long_description=readme(),
        zip_safe=True)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
