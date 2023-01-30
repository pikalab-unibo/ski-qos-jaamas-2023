import distutils.cmd
import itertools
import pandas as pd
from psyki.logic.prolog import TuProlog
from psyki.ski import Injector
from setuptools import setup, find_packages
from datasets import load_splice_junction_dataset, load_breast_cancer_dataset, load_census_income_dataset, \
    SpliceJunction, BreastCancer, CensusIncome
from experiment import grid_search, create_nn, create_educated_nn, NEURONS_PER_LAYERS, LAYERS, filter_neurons
from knowledge import generate_missing_knowledge, PATH as KNOWLEDGE_PATH
from results import PATH as RESULT_PATH


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
        splice_train, splice_test = load_splice_junction_dataset(self.binary_f, self.numeric_out)
        splice_train.to_csv(SpliceJunction.file_name, index=False)
        splice_test.to_csv(SpliceJunction.file_name_test, index=False)

        breast_train, breast_test = load_breast_cancer_dataset(self.binary_f, self.numeric_out)
        breast_train.to_csv(BreastCancer.file_name, index=False)
        breast_test.to_csv(BreastCancer.file_name_test, index=False)

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


class FindBestConfiguration(distutils.cmd.Command):
    description = 'Search for best predictor\'s parameters w.r.t. accuracy'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        datasets = [BreastCancer, SpliceJunction, CensusIncome]
        injectors = [Injector.kins, Injector.kill]
        injector_names = ['kins', 'kill']
        for dataset in datasets:
            params = {
                'neurons': list(list(x) for x in itertools.product(NEURONS_PER_LAYERS, repeat=len(LAYERS))),
                'hidden_layers': LAYERS
            }
            print("\n\nGrid search for predictors for the " + dataset.name + " dataset")
            best_params = {'uneducated': grid_search(dataset.name, params, create_nn)}
            new_neurons = [filter_neurons(x) for x in best_params['uneducated']['neurons']]
            max_layers = best_params['uneducated']['hidden_layers']
            new_params = {
                'neurons': list(list(x) for x in itertools.product(*new_neurons)),
                'hidden_layers': list(range(1, max_layers+1))
            }
            data = {'uneducated': best_params['uneducated']['neurons']}
            for injector, injector_name in zip(injectors, injector_names):
                print("\n" + injector_name)
                new_params['injector'] = [injector]
                new_params['formulae'] = [TuProlog.from_file(KNOWLEDGE_PATH / dataset.knowledge_file_name).formulae]
                best_params[injector_name] = grid_search(dataset.name, new_params, create_educated_nn)
                data[injector_name] = best_params[injector_name]['neurons']
            pd.DataFrame(data).to_csv(RESULT_PATH / (dataset.name + '.csv'))


class RunExperiments(distutils.cmd.Command):
    description = 'Run experiments, a.k.a. compute metrics'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        # for d in dataset:
        #   for i in injectors:
        #       ...
        #       compute_metrics(uneducated, educated, params)

        census_income_knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "census-income.pl")).formulae
        splice_junction_knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "splice-junction.pl")).formulae
        breast_cancer_knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "breast-cancer.pl")).formulae
        # TODO: complete with the code for the experiments
        # This is just a sketch to ensure that the knowledge is properly loaded.
        print('\n\n' + 25 * '-' + ' Census Income Knowledge ' + 25 * '-' + '\n\n')
        for rule in census_income_knowledge:
            print(rule)
        print('\n\n' + 25 * '-' + ' Splice Junction Knowledge ' + 25 * '-' + '\n\n')
        for rule in splice_junction_knowledge:
            print(rule)
        print('\n\n' + 25 * '-' + ' Breast Cancer Knowledge ' + 25 * '-' + '\n\n')
        for rule in breast_cancer_knowledge:
            print(rule)


setup(
    name='SKI QoS',  # Required
    description='SKI QoS experiments',
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
        'psyki>=0.2.15.dev2',
        'psyke>=0.3.3.dev13',
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
        'grid_search': FindBestConfiguration,
    },
)
