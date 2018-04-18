from setuptools import setup, find_packages

setup(
    name='keras_svm',
    version='1.0.0b8',
    description='A model to use keras models with Support Vector Machines',
    url='https://github.com/Luttik/keras_svm/tree/master',  # Optional
    author='Daan Luttik',  # Optional
    author_email='d.t.luttik@gmail.com',  # Optional
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='keras sklearn svm ml',  # Optional
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    install_requires=['keras', 'scikit-learn'],  # Optional
)
