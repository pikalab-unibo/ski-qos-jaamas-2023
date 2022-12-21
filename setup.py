import distutils.cmd
from setuptools import setup, find_packages
from datasets import load_splice_junction_dataset, load_breast_cancer_dataset, load_census_income_dataset, \
    SpliceJunction, BreastCancer, CensusIncome


class LoadDatasets(distutils.cmd.Command):
    description = 'download necessary datasets for the experiments'
    user_options = [('features=', 'f', 'binarize the features of the datasets (y/[n])'),
                    ('output=', 'o', 'convert class string name into numeric indices (y/[n])')]
    binary_f = False
    numeric_out = False
    features = 'n'
    output = 'n'

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        self.binary_f = self.features.lower() == 'y'
        self.numeric_out = self.output.lower() == 'y'

    def run(self) -> None:
        load_splice_junction_dataset(self.binary_f, self.numeric_out).to_csv(SpliceJunction.file_name, index=False)
        load_breast_cancer_dataset(self.binary_f, self.numeric_out).to_csv(BreastCancer.file_name, index=False)
        load_census_income_dataset(self.binary_f, self.numeric_out).to_csv(CensusIncome.file_name, index=False)


setup(
    name='kins',  # Required
    description='KINS knowledge injection algorithm test',
    license='Apache 2.0 License',
    url='https://github.com/MatteoMagnini/kins-experiments',
    author='Matteo Magnini',
    author_email='matteo.magnini@unibo.it',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='symbolic knowledge injection, ski, symbolic ai',  # Optional
    # package_dir={'': 'src'},  # Optional
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.9.0, <3.10',
    install_requires=[
        'psyki>=0.2.10',
        'psyke>=0.3.2.dev8',
        'tensorflow>=2.7.0',
        'numpy>=1.22.3',
        'scikit-learn>=1.0.2',
        'pandas>=1.4.2',
    ],  # Optional
    zip_safe=False,
    cmdclass={
        'load_datasets': LoadDatasets,
    },
)
