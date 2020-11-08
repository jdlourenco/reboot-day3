from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'category_encoders',
    'joblib',
    'pytz',
    'gcsfs==0.6.0',
    's3fs',
    'pandas',
    'scikit-learn==0.20.4',
    'pygeohash',
    'category_encoders',
    'termcolor',
    'mlflow',
    'xgboost',
    'memoized_property',
    'psutil']

setup(
    name='TaxiFareModel',
    version='1.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Taxi Fare Prediction Pipeline'
)




