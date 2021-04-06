from setuptools import setup

with open('requirements.txt') as file:
    requirements = [line.strip() for line in file.readlines()]

setup(
    name='python-lasvm',
    version='0.1',
    description='A Python implementation of the LaSVM online learning algorithm.',
    url='https://github.com/ylytkin/python-lasvm',
    author='Yura Lytkin',
    author_email='jurasicus@gmail.com',
    license='MIT',
    packages=['lasvm'],
    install_requires=requirements,
    zip_safe=False,
)
