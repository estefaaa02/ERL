import os
from setuptools import setup

def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('requirements.txt').splitlines()

setup(name='ERL',
      version='1.0',
      description='ERL: Emotion Recognition Library',
      url='https://github.com/estefaaa02/ERL',
      author='Mario Gómez Estefanía Pérez Victoria Núñez',
      author_email='mgomezcam@unbosque.edu.co eperezt@unbosque.edu.co vnunezd@unbosque.edu.co',
      license='Apache License, Version 2.0',
      packages=['ERL'],
      package_data={
        'ERL': ['models/*']
      },
      zip_safe=False,
      install_requires=requirements,
      )