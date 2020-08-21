from distutils.core import setup

setup(name='OptiDiff',
      version='1.0',
      description='Structural variation detection tool using optical map data',
      author='Mehmet Akdel',
      author_email='mehmet.akdel@wur.nl',
      url='https://gitlab.com/akdel/',
      packages=['OptiDiff'],
      install_requires=["scipy", "numba", "intervaltree", "matplotlib", "fire"])
