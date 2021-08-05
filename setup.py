from distutils.core import setup

setup(
    name = 'pupillometer',
    version = '0.0.1',
    packages = ['pupillometer'],
    description = 'Pupil diameter tracker',
    author = 'Leo Molina',
    author_email = 'leonardomt@gmail.com',
    url = 'https://github.com/leomol/pupil-tracker',
    download_url = 'https://github.com/leomol/pupil-tracker/archive/refs/tags/v0.0.1.tar.gz',
    keywords = ['pupillometer', 'pupillometry', 'pupil', 'tracker'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8'
    ],
)