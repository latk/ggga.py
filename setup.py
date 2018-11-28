from setuptools import setup  # type: ignore


def parse_requirements(filename):
    reqs = []
    with open(filename) as reqfile:
        for line in reqfile:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            reqs.append(line)
    return reqs


setup(
    name='ggga',
    python='>=3.6',
    install_requires=parse_requirements('./requirements.txt'),
    extras_require={
        'dev': parse_requirements('./requirements-dev.txt'),
    },
)
