from setuptools import find_packages, setup

setup(
    name="polycim",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "polycim=polycim.cli.main:main",
        ],
    },
    install_requires=[
        # 在这里列出你的依赖包
    ],
)
