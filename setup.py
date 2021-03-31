import setuptools

REQUIRED_PACKAGE = [
    'torch>=1.7.0',
]

setuptools.setup(
    name='saja',
    project_name="self-attention-jet-assignment",
    version="0.0.1",
    author="Seungjin Yang",
    author_email="seungjin.yang@cern.ch",
    description="SW for https://arxiv.org/abs/2012.03542",
    url="https://github.com/seungjin-yang/SaJa",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=REQUIRED_PACKAGE
)
