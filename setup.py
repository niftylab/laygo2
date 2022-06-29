# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='laygo2',
    version='0.1.0',
    description='LAYout with Gridded Object 2 - A Python Package for Scripted Layout Generation Flow',
    long_description='readme',
    author='Jaeduk Han',
    url='',
    license=license,
    package=find_packages(exclude=('test', 'docs'))
)