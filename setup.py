from setuptools import find_packages, setup
HYPHEN_E_DOT = '-e .'

def get_requirements(filepath):
    with open(filepath) as file:
        requirements = [req.replace("\n", "") for req in file.readlines()]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        return requirements

setup(
    name='maternal_health_risk_predictor',
    version='0.0.1',
    author='Aleena',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)