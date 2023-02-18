import os
from setuptools import setup
from mapchete import __version__, __author__, __author_email__


def read(fname, read_type="read"):
    assert read_type in {"read", "readlines"}
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        file = f.read() if read_type == "read" else f.readlines()
    return file


setup(
    name="mapchete",
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    description=("Croping geospatial data for deep learning purposed maximizing images spatial distribution"),
    license="Apache 2",
    url="https://github.com/abetatos/mapchete",
    packages=['mapchete'],
    install_requires=read("requirements.txt", "readlines"),
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: Beta",
        "Topic :: Geospatial data",
        "License :: Apache 2",
    ],
)