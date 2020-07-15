from setuptools import setup, find_packages

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
      zip_safe=False)