from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
      requirements = [req.strip() for req in f.readlines()]

setup(
    name = 'my_utils',
    description = 'Useful bricks of code for data exploration',
    author_email= 'chl.charlesleclerc@gmail.com',
    author = 'Charles Leclerc',
    packages = find_packages(),
    install_requires=requirements
    )
