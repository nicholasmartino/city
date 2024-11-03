from setuptools import setup, find_packages

setup(
    name='city',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here
        ],
    },
    author='Nicholas Martino',
    author_email='nicholas.martino@hotmail.com',
    description='Object model for urban elements',
    url='https://github.com/nicholasmartino/city',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)