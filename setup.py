from setuptools import setup, find_packages

VERSION = '0.0.13'
DESCRIPTION = 'Images to Power spectrum Pipeline'

with open("README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name = "pipy"
    version = VERSION
    authors = [{name="Yuchen Liu", email="yl871@cam.ac.uk"}]
    description = DESCRIPTION
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages()
    readme = "README.md"
    url = "https://github.com/ycliu23/pipy"
    keywords=["python","astrophysics","cosmology","cosmic dawn","reionization","power spectrum"],
    classifiers =
        {
            'License :: OSI Approved :: MIT License'
            'Programming Language :: Python :: 3.7'
            'Programming Language :: Python :: 3.8'
            'Programming Language :: Python :: 3.9'
        }
    )
