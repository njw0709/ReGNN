import setuptools

# reading long description from file
with open("README.md") as file:
    long_description = file.read()


# specify requirements of your package here
REQUIREMENTS = [
    "torch",
    "pandas",
    "artemis",
    "scikit-learn",
    "numpy",
    "pyro-ppl",
    "stata-setup",
    "ray",
    "alibi",
    "matplotlib",
]

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
    description="hybrid-NN-linear regression model that captures all-way interactions",
    long_description=long_description,
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=REQUIREMENTS,
    keywords="interactions effect, hybrid model, shap, pytorch, linear regression, neural network, explainable AI",
)
