from setuptools import setup

setup(
    name='ggga',
    install_requires=[
        'attrs>=18.0',
    ],
    extras_require={
        'dev': [
            'flake8',
            'pylint',
            'pytest',
            'pytest-describe',
        ]
    },
)
