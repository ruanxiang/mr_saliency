"""setup for mr_saliency to enable pip install"""

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
        url='https://github.com/ruanxiang/mr_saliency',
        author='ruanxiang (xkunglu added installer and py3 port)',
        license='GPL2',
        dependency_links=['wxPython'],
        install_requires=['opencv-python', 'numpy',
                          'scikit-image', 'matplotlib', 'scipy'],
        packages=find_packages(),
        long_description=readme(),
        zip_safe=True)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
