#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from distutils.core import setup
import numpy as np

ext_modules = []
cmdclass = {}
    name='uimnet',
    version='0.0.0',
    description='',
    url='http://github.com/ishmaelbelghazi/uimnet',
    author='Mohamed Ishmael Belghazi and David Lopez-Paz',
    author_email='ishmael.belghazi@gmail.com',
    license='MIT',
    packages=['uimnet'],
    cmdclass=cmdclass,

    ext_modules=ext_modules,
)
