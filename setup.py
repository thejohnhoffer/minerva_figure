import os
from configparser import ConfigParser
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    README = f.read()

requires = [
    'numpy>=1.11.1',
    'pyaml>=16.12.2',
    'opencv-python>=3.3.0.10',
]

def read_version():
    config = ConfigParser()
    config.read('setup.cfg')
    return config.get('metadata','version')


VERSION = read_version()
DESCRIPTION = 'minerva scripts'
AUTHOR = 'D.P.W. Russell'
LICENSE = 'AGPL-3.0'
HOMEPAGE = 'https://github.com/thejohnhoffer/minerva-scripts'

setup(
    name='minerva_scripts',
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
    entry_points={
        'console_scripts': [
            'combine=minerva_scripts.scripts.combine:main',
        ]
    },
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    author=AUTHOR,
    author_email='douglas_russell@hms.harvard.edu',
    license=LICENSE,
    url=HOMEPAGE,
    download_url='%s/archive/v%s.tar.gz' % (HOMEPAGE, VERSION),
    keywords=['minerva', 'scripts', 'microscopy'],
    zip_safe=False,
)
