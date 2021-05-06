import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setuptools.setup(
    name="optimal_stopping",
    version="0.0.1",
    author="C. Herrera, F. Krach, P. Ruyssen, J. Teichmann",
    author_email="calypso.herrera@math.ethz.ch, florian.krach@me.com",
    description="Code used to run experiments for paper 'Optimal Stopping via "
                "Randomized Neural Networks'.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HeKrRuTe/OptStopRandNN",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
