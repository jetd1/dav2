# setup the module
import setuptools

def packages():
    return setuptools.find_packages()


setuptools.setup(
    name="depth_anything_v2",
    version="0.0.1",
    author="xxxx",
    author_email="xxxx@xxx.xx",
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    packages=packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='~=3.9',
    install_requires=[
        'opencv-python-headless>=4',
        'xformers==0.0.28.post2'
    ],
    entry_points=dict()
)
