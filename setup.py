# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='laygo2',
    version='0.1.0',
    author='Jaeduk Han',
    description='LAYout with Gridded Object 2 - A Python Package for Scripted IC Layout Generation Flow',
    long_description=readme,
    url='https://github.com/niftylab/laygo2', # project address
    license=license,
    packages=find_packages(exclude=('test', 'docs')),
    python_requires='>=3.0',
)
