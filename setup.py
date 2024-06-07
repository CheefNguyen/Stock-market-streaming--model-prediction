from setuptools import setup, find_packages

setup(
    name='thesis_app',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask', 'kafka-python', 'requests', 'pymongo', 'python-dotenv','numpy',
        'pandas',
        'torch',
        'tensorflow',
        # Add any other dependencies here
    ],
)