from setuptools import find_packages, setup

setup(
    name='ja_pk',
    packages=find_packages("src"),
    version='0.1.0',
    description='Various Python functions for use in data science projects',
    author='jonathan',
    license='',
    package_dir={"": "src"},
    install_requires=[]
)
