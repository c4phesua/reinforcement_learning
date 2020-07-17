from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    lse = f.read()

setup(
    name='DeepLearning',
    version='0.1',
    packages=['rf_model'],
    url='https://github.com/th1590/reinforcement_learning',
    license=lse,
    author='Carokann',
    author_email='th1475369@gmail.com',
    description='reinforcement learning algorithms',
    long_description=readme,
)
