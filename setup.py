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
    LOSS
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
        # Iterate over datasets
        for dataset in datasets:
            print('\n\n' + dataset.name)
            results_training = pd.DataFrame(columns=columns)
            results_inference = pd.DataFrame(columns=columns)
            train_data = pd.read_csv(dataset.file_name)
            test_data = pd.read_csv(dataset.file_name_test)
            best_params = pd.read_csv(RESULT_PATH / (dataset.name + '.csv'), index_col=0)
            uneducated_neurons = eval(best_params.loc['uneducated']['neurons'])
            layers = len(uneducated_neurons)
            # Iterate over injectors
            for injector, injector_name in zip(injectors, injector_names):
                print(injector_name)
                if injector_name != 'kbann':
                    educated_neurons = eval(best_params.loc[injector_name]['neurons'])
                else:
                    educated_neurons = uneducated_neurons
                formulae = TuProlog.from_file(KNOWLEDGE_PATH / dataset.knowledge_file_name).formulae
                for formula in formulae:
                    formula.trainable = True
                # KILL injector needs the class mapping
                injection_params = {'feature_mapping': {k: v for v, k in enumerate(train_data.columns)}}
                if injector_name == 'kill':
                    injection_params['class_mapping'] = dataset.class_mapping_short
                # Create the educated predictor
                set_seed(SEED)
                educated = create_educated_nn(len(train_data.columns) - 1, len(dataset.class_mapping_short), layers,
                                              educated_neurons, injector, formulae, injection_params)
                for metric in metrics_names:
                    params = {
                        'x': train_data.iloc[:, :-1],
                        'y': train_data.iloc[:, -1:],
                        'epochs': EPOCHS,
                        'batch_size': BATCH_SIZE,
                        'verbose': VERBOSE,
                        'metric': metric
                    }

                    # Create the uneducated predictor
                    set_seed(SEED)
                    uneducated = create_nn(len(train_data.columns) - 1, len(dataset.class_mapping), layers, uneducated_neurons)
                    if metric != 'data efficiency':
                        results_training.loc[injector_name, metric] = compute_metrics_training(uneducated, educated, params)
                    else:
                        # Compute accuracy over POPULATION_SIZE (30) runs to get a statistically significant result
                        accuracies_educated = []
                        accuracies_uneducated = []
                        for idx in range(POPULATION_SIZE):
                            print("Training and computing accuracy for run " + str(idx + 1) + " of " + str(POPULATION_SIZE) + "...")
                            set_seed(SEED + idx)
                            uneducated = create_nn(len(train_data.columns) - 1, len(dataset.class_mapping), layers, uneducated_neurons)
                            uneducated.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1:], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=CALLBACK)
                            accuracies_uneducated.append(uneducated.evaluate(test_data.iloc[:, :-1], test_data.iloc[:, -1:])[1])
                            set_seed(SEED + idx)
                            educated = create_educated_nn(len(train_data.columns) - 1, len(dataset.class_mapping_short), layers, educated_neurons, injector, formulae, injection_params)
                            educated.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1:], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=CALLBACK)
                            if injector_name == 'kill' and isinstance(educated, LambdaLayer.ConstrainedModel):
                                educated = educated.remove_constraints()
                                educated.compile(optimizer='adam', loss=LOSS, metrics=['accuracy'])
                            accuracies_educated.append(educated.evaluate(test_data.iloc[:, :-1], test_data.iloc[:, -1:])[1])

                        accuracy_history = pd.DataFrame({'uneducated': accuracies_uneducated, 'educated': accuracies_educated})
                        accuracy_history.to_csv(RESULT_PATH / (dataset.name + '_' + injector_name + '_accuracy.csv'))
                        params['train_x1'] = train_data.iloc[:, :-1]
                        params['train_y1'] = train_data.iloc[:, -1:]
                        params['train_x2'] = train_data.iloc[:, :-1]
                        params['train_y2'] = train_data.iloc[:, -1:]
                        params['epochs1'] = EPOCHS
                        params['epochs2'] = EPOCHS
                        params['metric1'] = np.mean(accuracies_uneducated)
                        params['metric2'] = np.mean(accuracies_educated)
                        params.pop('x')
                        params.pop('y')
                        params.pop('epochs')

                    params['metric'] = metric
                    # KILL injector needs the constraints to be removed
                    if injector_name == 'kill' and isinstance(educated, LambdaLayer.ConstrainedModel):
                        educated = educated.remove_constraints()
                        educated.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    if metric == 'data efficiency':
                        params['metric'] = metric
                        data_efficiency = compute_metrics_inference(uneducated, educated, params)
                        results_training.loc[injector_name, metric] = data_efficiency
                        results_inference.loc[injector_name, metric] = results_training.loc[injector_name, metric]
                        results_training.loc[injector_name, 'accuracy'] = params['metric2']
                        results_inference.loc[injector_name, 'accuracy'] = params['metric2']
                        results_training.loc[injector_name, 'accuracy uneducated'] = params['metric1']
                        results_inference.loc[injector_name, 'accuracy uneducated'] = params['metric1']
                    else:
                        params.pop('y')
                        params.pop('epochs')
                        params.pop('batch_size')
                        results_inference.loc[injector_name, metric] = compute_metrics_inference(uneducated, educated, params)

            results_training.to_csv(RESULT_PATH / (dataset.name + '_metrics_training.csv'))
            results_inference.to_csv(RESULT_PATH / (dataset.name + '_metrics_inference.csv'))


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
