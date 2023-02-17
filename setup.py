import distutils.cmd
import itertools
import numpy as np
import pandas as pd
from psyki.logic.prolog import TuProlog
from psyki.ski import Injector
from psyki.ski.kill import LambdaLayer
from setuptools import setup, find_packages
from tensorflow.python.framework.random_seed import set_seed
from datasets import load_splice_junction_dataset, load_breast_cancer_dataset, load_census_income_dataset, \
    SpliceJunction, BreastCancer, CensusIncome
from experiment import grid_search, create_nn, create_educated_nn, NEURONS_PER_LAYERS, LAYERS, filter_neurons, \
    compute_metrics_training, compute_metrics_inference, EPOCHS, BATCH_SIZE, VERBOSE, SEED, CALLBACK, POPULATION_SIZE, \
    LOSS, run_experiments
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
            indices = ['uneducated']
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
                'hidden_layers': list(range(1, max_layers + 1)),
                'accuracy': best_params['uneducated']['accuracy']
            }
            data = {'neurons': [best_params['uneducated']['neurons']],
                    'accuracy': [best_params['uneducated']['accuracy']]}
            for injector, injector_name in zip(injectors, injector_names):
                print("\n" + injector_name)
                indices.append(injector_name)
                new_params['injector'] = [injector]
                new_params['formulae'] = [TuProlog.from_file(KNOWLEDGE_PATH / dataset.knowledge_file_name).formulae]
                best_params[injector_name] = grid_search(dataset.name, new_params, create_educated_nn)
                data['neurons'].append(best_params[injector_name]['neurons'])
                data['accuracy'].append(best_params[injector_name]['accuracy'])
            pd.DataFrame(data, index=indices).to_csv(RESULT_PATH / (dataset.name + '.csv'))


class RunExperiments(distutils.cmd.Command):
    description = 'Run experiments, a.k.a. compute metrics'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        datasets = [BreastCancer, SpliceJunction, CensusIncome]
        injectors = [Injector.kins, Injector.kill, Injector.kbann]
        injector_names = ['kins', 'kill', 'kbann']
        metrics_names = ['energy', 'memory', 'latency', 'data efficiency']
        columns = metrics_names + ['accuracy']
        run_experiments(datasets, injectors, injector_names, metrics_names, columns)


class UpdateExperiments(distutils.cmd.Command):
    """A custom command to update all metrics but data efficiency."""

    description = 'Update all metrics but data efficiency'
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        datasets = [BreastCancer, SpliceJunction, CensusIncome]
        injectors = [Injector.kins, Injector.kill, Injector.kbann]
        injector_names = ['kins', 'kill', 'kbann']
        metrics_names = ['energy', 'memory', 'latency']
        # Load old results
        run_experiments(datasets, injectors, injector_names, metrics_names, metrics_names, update=True)


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
        'psyki>=0.2.19',
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
        'update_experiments': UpdateExperiments,
    },
)
