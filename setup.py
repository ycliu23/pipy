from setuptools import setup

VERSION = '0.0.13'
DESCRIPTION = 'Images to Power spectrum Pipeline'
LONG_DESCRIPTION = 'Python package that allows generation of 3D, 2D and 1D power spectra from images'

setup(
    name = "pipy"
    version = VERSION
    authors = [{name="Yuchen Liu", email="yl871@cam.ac.uk"}]
    description = DESCRIPTION
    long_description=long_description,
    long_description_content_type="text/markdown",
    readme = "README.md"
    url = "https://github.com/ycliu23/pipy"
    classifiers =
        {
            'License :: OSI Approved :: MIT License'
            'Programming Language :: Python :: 3.7'
            'Programming Language :: Python :: 3.8'
            'Programming Language :: Python :: 3.9'
        }
    )
