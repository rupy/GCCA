from setuptools import setup, find_packages

setup(
    name='gcca',
    description='Generalized Canonical Correlation Analysis library',
    author='rupy',
    author_email='rupyapps@gmail.com',
    license='MIT License',
    version='0.1dev',
    url='https://github.com/rupy/GCCA',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py'],
    zip_safe=False,
    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
    ],
)