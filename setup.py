from setuptools import setup

_packages = \
[
    "mrarbdcf",
]

_package_dir = \
{
    "mrarbdcf":"./mrarbdcf_src/", 
}

setup\
(
    name = 'mrarbdcf',
    packages = _packages,
    package_dir = _package_dir,
    include_package_data = True
)
