# -*- coding: utf-8 -*-
"""setup.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RW7lI1qwOvCnVQwyytt95wfN99x-qzIU
"""

!pip install min-dalle
import setuptools
# from pathlib import Path

# Check if running in a notebook environment
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        # If in a notebook, skip setup() execution
        print("Skipping setuptools.setup() in notebook environment.")
    else:
        # If not in a notebook, execute setup()
        setuptools.setup(
            name='min-dalle',
            description='min(DALL·E)',
            # long_description=(Path(__file__).parent / "README.rst").read_text(),
            version='0.4.11',
            author='Brett Kuprel',
            author_email='brkuprel@gmail.com',
            url='https://github.com/kuprel/min-dalle',
            packages=[
                'min_dalle',
                'min_dalle.models'
            ],
            license='MIT',
            install_requires=[
                'torch>=1.11',
                'typing_extensions>=4.1',
                'numpy>=1.21',
                'pillow>=7.1',
                'requests>=2.23',
                'emoji'
            ],
            keywords=[
                'artificial intelligence',
                'deep learning',
                'text-to-image',
                'pytorch'
            ]
        )
except ImportError:
    # If IPython is not available, assume not in a notebook and execute setup()
    setuptools.setup(
        name='min-dalle',
        description='min(DALL·E)',
        # long_description=(Path(__file__).parent / "README.rst").read_text(),
        version='0.4.11',
        author='Brett Kuprel',
        author_email='brkuprel@gmail.com',
        url='https://github.com/kuprel/min-dalle',
        packages=[
            'min_dalle',
            'min_dalle.models'
        ],
        license='MIT',
        install_requires=[
            'torch>=1.11',
            'typing_extensions>=4.1',
            'numpy>=1.21',
            'pillow>=7.1',
            'requests>=2.23',
            'emoji'
        ],
        keywords=[
            'artificial intelligence',
            'deep learning',
            'text-to-image',
            'pytorch'
        ]
    )