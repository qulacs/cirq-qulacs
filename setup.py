from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='cirqqulacs',
    version='0.0.1',
    author='QunaSys',
    author_email='qulacs@qunasys.com',
    url='http://www.qulacs.org',
    description='Fast quantum circuit simulator Qulacs as backend for Cirq',
    long_description=readme,
    license=license,
    packages=find_packages(exclude=['tests']),
    test_suite = 'tests'
)
