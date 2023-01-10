from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


PATH = Path(__file__).parents[0]
UCI_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


class SpliceJunction(object):
    data_url: str = UCI_URL + "molecular-biology/splice-junction-gene-sequences/splice.data"
    file_name: str = PATH / "splice-junction-data.csv"
    features: list[str] = ['X' + (str(i) if i > 0 else "_" + str(abs(i))) for i in list(range(-30, 0)) + list(range(1, 31))]
    features_values: list[str] = ['a', 'c', 'g', 't']
    sub_features: list[str] = [f + v for f in features for v in ['a', 'c', 'g', 't']]  # Can't reference features_values
    class_mapping: dict[str, int] = {'exon-intron': 0, 'intron-exon': 1, 'none': 2}
    class_mapping_short: dict[str, int] = {'ei': 0, 'ie': 1, 'n': 2}
    aggregate_features: dict[str, tuple[str]] = {'a': ('a',), 'c': ('c',), 'g': ('g',), 't': ('t',),
                                                 'd': ('a', 'g', 't'), 'm': ('a', 'c'), 'n': ('a', 'c', 'g', 't'),
                                                 'r': ('a', 'g'), 's': ('c', 'g'), 'y': ('c', 't')}


class BreastCancer(object):
    data_url: str = UCI_URL + "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    file_name: str = PATH / "breast-cancer-data.csv"
    features: list[str] = ["ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
                           "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses",
                           "diagnosis"]
    class_mapping: dict[str, int] = {'benign': 0, 'malignant': 1}
    class_mapping_short: dict[str, int] = {'b': 0, 'm': 1}


class CensusIncome(object):
    data_url: str = UCI_URL + "adult/adult.data"
    data_test_url: str = UCI_URL + "adult/adult.test"
    file_name: str = PATH / "census-income-data.csv"
    file_name_test: str = PATH / "census-income-data-test.csv"
    features: list[str] = ["Age", "WorkClass", "FinalWeight", "Education", "EducationNumeric", "MaritalStatus",
                           "Occupation", "Relationship", "Ethnicity", "Sex", "CapitalGain", "CapitalLoss",
                           "HoursPerWeek", "NativeCountry", "income"]
    categorical_features: list[str] = ['Education', 'Ethnicity', 'MaritalStatus', 'NativeCountry', 'Occupation',
                                       'Relationship', 'Sex', 'WorkClass']
    class_mapping: dict[str, int] = {'<=50K': 0, '>50K': 1}


def load_splice_junction_dataset(binary_features: bool = False, numeric_output: bool = False) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(SpliceJunction.data_url, sep=r",\s*", header=None, encoding='utf8')
    df.columns = ["class", "origin", "DNA"]

    def binarize_features(df_x: pd.DataFrame, mapping: dict[str: set[str]]) -> pd.DataFrame:
        def get_values() -> Iterable[str]:
            result = set()
            for values_set in mapping.values():
                for v in values_set:
                    result.add(v)
            return result

        sub_features = sorted(get_values())
        results = []
        for _, r in df_x.iterrows():
            row_result = []
            for value in r:
                positive_features = mapping[value]
                for feature in sub_features:
                    row_result.append(1 if feature in positive_features else 0)
            results.append(row_result)
        return pd.DataFrame(results, dtype=int)

    # Split the DNA sequence
    x = []
    for _, row in df.iterrows():
        label, _, features = row
        features = list(f for f in features.lower())
        features.append(label.lower())
        x.append(features)
    df = pd.DataFrame(x)
    class_mapping = SpliceJunction.class_mapping_short
    if numeric_output:
        new_y = df.iloc[:, -1:].applymap(lambda y: class_mapping[y] if y in class_mapping.keys() else y)
    else:
        new_y = df.iloc[:, -1]
    if binary_features:
        # New binary sub features
        new_x = binarize_features(df.iloc[:, :-1], SpliceJunction.aggregate_features)
        new_y.columns = [new_x.shape[1]]
        df = new_x.join(new_y)
        df.columns = SpliceJunction.sub_features + ['class']
    else:
        # Original dataset
        df.iloc[:, -1] = new_y
        df.columns = SpliceJunction.features + ['class']
    return df


def load_breast_cancer_dataset(binary_features: bool = False, numeric_output: bool = False) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(BreastCancer.data_url, sep=",", header=None, encoding='utf8').iloc[:, 1:]
    df.columns = BreastCancer.features
    df.diagnosis = df.diagnosis.apply(lambda x: 0 if x == 2 else 1) if numeric_output else df.diagnosis
    df.BareNuclei = df.BareNuclei.apply(lambda x: 0 if x == '?' else x).astype(int)
    if binary_features:
        # One hot encode columns to make it works for KBANN
        new_df = []
        for column in df.columns[:-1]:
            new_df.append(pd.get_dummies(df[column]))
        new_df.append(df.diagnosis)
        new_df = pd.concat(new_df, axis=1)
        new_columns = [i + str(j) for i in df.columns[:-1] for j in list(range(1, 11))]
        new_columns.insert(50, "BareNuclei0")  # For missing values
        new_columns.remove("Mitoses9")  # Always absent
        new_columns.append("diagnosis")
        new_df.columns = new_columns
        df = new_df
    return df


def load_census_income_dataset(binary_features: bool = False, numeric_output: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    df: pd.DataFrame = pd.read_csv(CensusIncome.data_url, sep=",", header=None, encoding='utf8')
    df.columns = CensusIncome.features
    df.income = df.income.apply(lambda x: 0 if x == ' <=50K' else 1) if numeric_output else df.income
    df_test: pd.DataFrame = pd.read_csv(CensusIncome.data_test_url, sep=",", header=None, encoding='utf8', skiprows=1)
    df_test.columns = CensusIncome.features
    df_test.income = df.income.apply(lambda x: 0 if x == ' <=50K.' else 1) if numeric_output else df.income

    def binarize(data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = data.applymap(lambda x: np.NaN if x == ' ?' else x)  # Missing values to NaN
        data[CensusIncome.categorical_features] = data[CensusIncome.categorical_features].astype('category')
        category_columns = data.select_dtypes(['category']).columns
        data[category_columns] = data[category_columns].apply(lambda x: x.cat.codes)  # Category to integers abd NaN to -1
        return data

    if binary_features:
        df = binarize(df)
        df_test = binarize(df_test)
    return df, df_test