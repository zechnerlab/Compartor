import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='compartor',
    version='1.1.0',
    packages=['compartor'],
    url='http://github.com/zechnerlab/Compartor',
    license='Simplified BSD License',

    description='Automatic moment equation generation for stochastic compartment populations',
    long_description=long_description,
    long_description_content_type="text/markdown",

    author='Tobias Pietzsch',
    author_email='tobias.pietzsch@gmail.com',

    python_requires='>=3',
    install_requires=['sympy>=1.6'],
)

