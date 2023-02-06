from setuptools import setup

setup(
    name='db_rsk_pred',
    version='0.1.0',
    author='gupo',
    packages=['db_rsk_pred'],
    install_requires=[ 'pandas', 'sklearn', 'lightgbm', 
                        'python-configuration', 'PyMySQL', 
                        'python-dateutil==2.8.2','pytz', 
                        'scipy', 'tqdm', 'numpy', 'regex',
                         'xlrd', 'matplotlib', 'sklearn', 'optuna']
)
