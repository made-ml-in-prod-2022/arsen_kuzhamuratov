from setuptools import find_packages, setup


setup(
    name="online_inference",
    packages=find_packages(),
    version="0.1.0",
    description="Online inference",
    author="Arsen Kuzhamuratov",
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "pandas",
        "matplotlib",
        "scipy",
        "sklearn",
        "numpy",
        "pyaml",
        "requests"
        ]
)