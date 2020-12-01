import pathlib
import setuptools


def get_pysimt_version():
    with open('pysimt/__init__.py') as f:
        s = f.read().split('\n')[0]
        if '__version__' not in s:
            raise RuntimeError('Can not detect version from pysimt/__init__.py')
        return eval(s.split(' ')[-1])


setuptools.setup(
    name='pysimt',
    version=get_pysimt_version(),
    description='A PyTorch framework for Simultaneous Neural Machine Translation',
    url='https://github.com/ImperialNLP/pysimt',
    author='Ozan Caglayan, Veneta Haralampieva, Julia Ive, Andy Li',
    author_email='o.caglayan@ic.ac.uk',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Operating System :: POSIX',
    ],
    keywords='nmt neural-mt simultaneous translation sequence-to-sequence deep-learning pytorch',
    python_requires='~=3.7',
    install_requires=[
        'numpy', 'tqdm', 'pillow',
        'torch', 'torchvision', 'sacrebleu>1.4.10',
    ],
    packages=setuptools.find_packages(),
    scripts=[str(p) for p in pathlib.Path('bin').glob('*')],
    zip_safe=False)
