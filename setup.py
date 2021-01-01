# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

from setuptools import setup, find_packages


install_requires = ['hyperspy>=1.6',
                    'matplotlib',
                    'numpy',
                    ]
tests_require = ['pytest']


setup(name='gpa',
      version='0.1',
      description='Geometrical Phase analysis (GPA)',
      url='',
      author='Eric Prestat',
      author_email='eric.prestat@gmail.com',
      package_data={
          'gpa': ['hyperspy_extension.yaml'],
          },
      entry_points={'hyperspy.extensions': 'gpa = gpa'},
      license='BSD-3-Clause',
      packages=find_packages(),
      python_requires='~=3.6',
      install_requires=install_requires,
      tests_require=tests_require,
      zip_safe=False
      )