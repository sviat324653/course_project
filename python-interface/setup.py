from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import numpy


project_root = os.path.dirname(os.path.abspath(__file__))
library_dir = project_root
library_name = "multivex"
package_name = "multivex"

import platform
runtime_library_dirs = []
if platform.system() == "Linux":
    runtime_library_dirs.append(os.path.join('$ORIGIN', '.'))


global_variables = {}
with open(os.path.join(project_root, package_name, "__init__.py")) as fp:
    exec(fp.read(), global_variables)
package_version = global_variables["__version__"]


extensions = [
    Extension(
        f"{package_name}.multivex",
        sources=[os.path.join(package_name, "multivex.pyx")],
        include_dirs=[os.path.join(project_root, package_name), numpy.get_include()],
        library_dirs=[os.path.join(library_dir, package_name)],
        libraries=[library_name],
        runtime_library_dirs=runtime_library_dirs if runtime_library_dirs else None,
        extra_compile_args=["-O3"]
    )
]

setup(
    name="multivex",
    version=package_version,
    packages=[package_name],
    package_data={
        package_name: ["libmultivex.so", "scan.h"],
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
    author="Sviatoslav",
    license="MIT",
    description="CUDA-accelerated primitives",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.8',
    zip_safe=False,
)
