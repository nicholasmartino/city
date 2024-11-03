from setuptools import setup, find_packages

setup(
    name='city',
    version='0.1.0.dev0',
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"city": ["py.typed"]},
    include_package_data=True,
    install_requires=[
        requirement.strip() for requirement in open('requirements.txt').readlines()
    ],
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