from setuptools import setup,find_packages

setup(
    name='db_rsk_pred',
    version='0.1.0',
    author='gupo',
    packages=find_packages(),  # ['db_rsk_pred'],  自动找到该路径下的所有package
    install_requires=['pandas', 'scikit-learn', 'lightgbm',
                      'python-configuration', 'PyMySQL',
                      'python-dateutil==2.8.2', 'pytz',
                      'scipy', 'tqdm', 'numpy', 'regex',
                      'xlrd', 'matplotlib', 'optuna', 'shap']
)

