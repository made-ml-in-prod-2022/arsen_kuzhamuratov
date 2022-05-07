from setuptools import find_packages, setup


setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Training staff",
    author="Arsen Kuzhamuratov",
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "pyyaml"
        ],
    license="MIT",
)
