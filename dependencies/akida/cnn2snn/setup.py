from os import path
from setuptools import setup

# Read the contents of the README file
directory = path.abspath(path.dirname(__file__))
with open(path.join(directory, 'README'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cnn2snn',
    version='2.2.2',
    description='Keras to Akida CNN Converter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Alvaro Moran',
    author_email='amoran@brainchip.com',
    url='https://doc.brainchipinc.com',
    license='Apache 2.0',
    packages=['cnn2snn', 'cnn2snn.transforms', 'cnn2snn.calibration'],
    entry_points={
        'console_scripts': [ 'cnn2snn = cnn2snn.cli:main' ]
    },
    install_requires=['tensorflow==2.7.0', 'keras==2.7.0',
        'akida==2.2.2'],
)
