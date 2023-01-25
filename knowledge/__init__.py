import re
from pathlib import Path
import pandas as pd
from psyke.utils.logic import pretty_theory
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from psyke import Extractor
from datasets import PATH as DATA_PATH


PATH = Path(__file__).parents[0]


MAX_FEATURES_IN_RULE: int = 20
MAX_RULES: int = 20


def generate_missing_knowledge():

    def generate_knowledge(data_name):
        data = pd.read_csv(DATA_PATH / (data_name + '-data.csv'))
        predictor = DecisionTreeClassifier(max_depth=MAX_FEATURES_IN_RULE, max_leaf_nodes=MAX_RULES)
        train, _ = train_test_split(data, random_state=0, train_size=0.5)
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
        predictor.fit(train_x, train_y)
        extractor = Extractor.cart(predictor, max_depth=MAX_FEATURES_IN_RULE, max_leaves=MAX_RULES)
        knowledge = extractor.extract(train)
        knowledge = pretty_theory(knowledge)
        # Substitute X =< N.5 with X < N
        knowledge = re.sub(r"([A-Z][a-zA-Z0-9_]*)[ ]=<[ ]([+-]?([0-9]*))[.]?[0-9]+", r"\g<1> < \g<2>", knowledge)
        # Substitute X > N.5 with X > N
        knowledge = re.sub(r"([A-Z][a-zA-Z0-9_]*)[ ]>[ ]([+-]?([0-9]*))[.]?[0-9]+", r"\g<1> > \g<2>", knowledge)
        # Substitute 0.0 with 0 and 1.0 with 1
        knowledge = re.sub(r"0.0", r"0", knowledge)
        knowledge = re.sub(r"1.0", r"1", knowledge)
        with open(PATH / (data_name + ".pl"), "w") as text_file:
            text_file.write(knowledge)

    generate_knowledge('census-income')
    generate_knowledge('breast-cancer')
