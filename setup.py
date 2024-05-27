from setuptools import find_packages, setup

setup(
    name='FIERL',
    packages=[package for package in find_packages() if package.startswith('safepo')],
    package_data={'safepo': ['py.typed', 'version.txt']},
    install_requires=[
        'joblib',
        'scipy',
        "torch >= 1.10.0",
        'tensorboard >= 2.8.0',
        "wandb >= 0.13.0",
        'pyyaml >= 6.0',
        'matplotlib >= 3.7.1',
        "seaborn >= 0.12.2",
        "pandas >=  1.5.3",
    ],
    description='FIERL',
    author='',
    url='https://github.com/davidesartor/FIERL',
    author_email='davide.sartor.4@studenti.unipd.it',
    keywords='Fault Detection'
        'Active Fault Diagnosis'
        'Reinforcement Learning',
    license='Apache License 2.0',
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)