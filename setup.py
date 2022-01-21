from setuptools import setup, find_packages

setup(
    name='pyclim NorESM',
    version='0.0.1',
    description='''Scripts and functions developed by the NorESM group at MetNo
                for analysis of both raw and cmorized NorESM and CMIP6 output. 
                These script are mainly used on the Betzy sigma2 computer and NIRD 
                data storage infrastructure.''',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(where='pyclim_noresm'),

    python_requires='>=3.7, <4',
    # Command line scripts
    entry_points={ 
        'console_scripts': [
            'find_models.py=pyclim_noresm.data_search.find_models:main',
            'get_missing_models.py=pyclim_noresm.data_search.get_missing_models:main'

        ],
    }
)