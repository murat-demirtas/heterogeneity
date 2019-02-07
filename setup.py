""""Package setup file"""

import distutils.core

distutils.core.setup(
    name='hbnm',
    version='1.0.1',
    packages=['', 'hbnm', 'hbnm.model', 'hbnm.model.params'],
    url='',
    license='',
    author='Murat Demirtas',
    author_email='murat.demirtas@yale.edu',
    description='Large-scale Biophysical Network Model'
                'Heterogeneous Dynamic Mean-field Approch.'
)
