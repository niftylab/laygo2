# -*- coding: utf-8 -*-

'''
Pypi update instruction:
1) update the version number in setup.py.
2) run 'python setup.py sdist bdist_wheel' to build whl.
3) run 'python -m twine upload dist/*'.
4) type username and password.

'''

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='laygo2',
    version='0.2.3',
    author='Jaeduk Han',
    description='LAYout with Gridded Object 2 - A Python Package for Scripted IC Layout Generation Flow',
    long_description=readme,
    url='https://github.com/niftylab/laygo2', # project address
    license=license,
    packages=find_packages(exclude=('test', 'docs')),
    package_data={'': ['interface/skill_export.il', 
                       'interface/magic_export.tcl', 
                       'interface/gds_default.layermap', 
                      ]},
    include_package_data=True,
    python_requires='>=3.0',
)
