from setuptools import find_namespace_packages, setup


setup(
    name="echo-forcing",
    version="0.1.0",
    description="Interactive long-video generation with scene memory",
    packages=find_namespace_packages(
        include=["pipeline*", "streaming*", "utils*", "wan*"]
    ),
)
