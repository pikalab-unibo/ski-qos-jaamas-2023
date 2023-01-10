import distutils.cmd
from setuptools import setup, find_packages
from datasets import load_splice_junction_dataset, load_breast_cancer_dataset, load_census_income_dataset, \
    SpliceJunction, BreastCancer, CensusIncome
from knowledge import get_census_income_knowledge, get_splice_junction_knowledge, \
    get_breast_cancer_knowledge, generate_missing_knowledge


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
        census_train, census_test = load_census_income_dataset(self.binary_f, self.numeric_out)
        census_train.to_csv(CensusIncome.file_name, index=False)
        census_test.to_csv(CensusIncome.file_name_test, index=False)


class GenerateMissingKnowledge(distutils.cmd.Command):
    description = 'Extract knowledge from the census income dataset'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        generate_missing_knowledge()


class RunExperiments(distutils.cmd.Command):
    description = 'Run experiments'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        census_income_knowledge = get_census_income_knowledge()
        splice_junction_knowledge = get_splice_junction_knowledge()
        poker_knowledge = get_breast_cancer_knowledge()
        # TODO: complete with the code for the experiments
        # This is just a sketch to ensure that the knowledge is properly loaded.
        print('\n\n' + 25 * '-' + ' Census Income Knowledge ' + 25 * '-' + '\n\n')
        for rule in census_income_knowledge:
            print(rule)
        print('\n\n' + 25 * '-' + ' Splice Junction Knowledge ' + 25 * '-' + '\n\n')
        for rule in splice_junction_knowledge:
            print(rule)
        print('\n\n' + 25 * '-' + ' Poker Knowledge ' + 25 * '-' + '\n\n')
        for rule in poker_knowledge:
            print(rule)


setup(
    name='Ski qos',  # Required
    description='SKI qos experiments',
    license='Apache 2.0 License',
    url='https://github.com/pikalab-unibo/ski-qos-jaamas-experiments-2022',
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
        'generate_missing_knowledge': GenerateMissingKnowledge,
        'run_experiments': RunExperiments,
    },
)
