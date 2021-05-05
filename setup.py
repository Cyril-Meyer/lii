import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

requirements = ["numpy", "tqdm"]

setuptools.setup(
    name="lii",
    version="1.0.1",
    author="Cyril Meyer",
    author_email="contact@cyrilmeyer.eu",
    description="Large Image Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cyril-Meyer/lii",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
)
