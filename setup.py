import setuptools

# reading long description from file
with open("README.md") as file:
    long_description = file.read()


# specify requirements of your package here
REQUIREMENTS = ["shap", "pandas", "torch"]

# some more details
CLASSIFIERS = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Development Status :: 4 - Beta",
]

# calling the setup function
# TODO: separate dev / test / deploy setup with options
setuptools.setup(
    name="mihm",
    version="0.0.1",
    description="hybrid-NN-linear regression model to get interactions effect",
    long_description=long_description,
    # url="https://github.com/njw0709/ShapeYV2",
    author="Jong Woo Nam",
    author_email="namj@usc.edu",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=REQUIREMENTS,
    keywords="interactions effect, hybrid model, shap, pytorch, linear regression, neural network",
)
