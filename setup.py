import setuptools
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setuptools.setup(
    # Keep the original namespaced distribution name used by this project
    name="tts-webui.songbloom",
    packages=setuptools.find_namespace_packages(),
    version="0.1.0",
    author="Your Name",
    description="SongBloom package for tts-webui",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rsxdalv/tts-webui.songbloom",
    include_package_data=True,
    # Ensure common resource file types included in package distributions
    package_data={
        # include any text, yaml, json files under the package
        "SongBloom": [
            "**/*.txt",
            "**/*.yaml",
            "**/*.yml",
            "**/*.json",
            "**/*.rep",
            "**/*.pickle",
        ]
    },
    scripts=[],
    install_requires=[
        # add runtime deps here if required
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
