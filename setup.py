from setuptools import setup, find_packages

setup(
    name='cirqqulacs',
    version='0.0.3',
    author='QunaSys',
    author_email='qulacs@qunasys.com',
    url='http://www.qulacs.org',
    description='Fast quantum circuit simulator Qulacs as backend for Cirq',
    long_description='',
    license='MIT',
    packages=find_packages(exclude=['tests']),
)
