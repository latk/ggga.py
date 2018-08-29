from setuptools import setup

setup(
    name='ggga',
    install_requires=[
        'attrs>=18.0',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-describe',
        ]
    },
)
