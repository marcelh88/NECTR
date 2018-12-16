import pandas as pd
import numpy as np
import random
from enum import Enum


class Sampling(Enum):
    UNIFORM = 1
    DISTANCE = 2
    DISTANCE_IN_CATEGORY = 3


class DataHelper:
    categories = None
    item2category = None
    category2item = None
    df_triple2id = None
    df_entity2id = None
    item_ids = None
    solution_ids = None
    df_relation2id = None
    item_distance = None
    data = None
    adjacency = None
    df_sol_mat = None
    sampling = Sampling.UNIFORM

    def __init__(self):
        pass

    @staticmethod
    def setup(data_path, parameters=None):
        DataHelper.categories = pd.read_csv(data_path + 'Categories.csv', sep=';')
        temp = DataHelper.categories.apply(lambda x: {int(i): x.name for i in x['entity_id'].split(',')},
                                    axis=1).values.tolist()
        DataHelper.item2category = {k: v for i in temp for k, v in i.items()}
        temp = DataHelper.categories.apply(lambda x: {x.name: list(map(int, x['entity_id'].split(',')))},
                                    axis=1).values.tolist()
        DataHelper.category2item = {k: v for i in temp for k, v in i.items()}
        temp = None

        DataHelper.df_triple2id = pd.read_table(data_path + 'triple2id.txt', index_col=None, header=0, names=['h', 't', 'r'])
        DataHelper.df_entity2id = pd.read_table(data_path + 'entity2id.txt', index_col=None, header=0,
                                     names=['entity', 'id', 'type'], encoding='latin1')
        DataHelper.item_ids = DataHelper.df_entity2id[DataHelper.df_entity2id['type'] == 'item']['id'].astype(int).tolist()
        DataHelper.solution_ids = DataHelper.df_entity2id[DataHelper.df_entity2id['type'] == 'solution']['id'].astype(int).tolist()
        DataHelper.df_relation2id = pd.read_table(data_path + 'relation2id.txt', index_col=None, header=0,
                                       names=['relation', 'id'])

        try:
            DataHelper.df_sol_mat = pd.read_csv(data_path + 'Solutions_matrix.csv', sep=';', index_col=0).T
        except FileNotFoundError:
            print('Solutions data in matrix format not found')
            print('Continuing...')

        DataHelper.set_parameters(parameters)

        if DataHelper.sampling == Sampling.DISTANCE or DataHelper.sampling == Sampling.DISTANCE_IN_CATEGORY:
            DataHelper.item_distance = np.load(data_path + 'item2item_distance.pickle')

        DataHelper.set_data(DataHelper.df_triple2id)

    @staticmethod
    def setup_prod_data(df_entity2id, df_relation2id):
        DataHelper.df_entity2id = df_entity2id
        DataHelper.item_ids = DataHelper.df_entity2id[DataHelper.df_entity2id['type'] == 'item']['id'].astype(int).tolist()
        DataHelper.solution_ids = DataHelper.df_entity2id[DataHelper.df_entity2id['type'] == 'solution']['id'].astype(int).tolist()
        DataHelper.df_relation2id = df_relation2id

    @staticmethod
    def set_parameters(parameters):
        if parameters:
            for p_name, p_value in parameters.items():
                setattr(DataHelper, p_name, p_value)

    @staticmethod
    def has_category(item):
        return item in DataHelper.item2category

    @staticmethod
    def get_category(item):
        return DataHelper.item2category[item]

    @staticmethod
    def get_category_encoding(item):
        encoding = [0] * DataHelper.get_category_count()
        if DataHelper.has_category(item):
            encoding[DataHelper.get_category(item)] = 1
        return encoding

    @staticmethod
    def get_categories():
        return DataHelper.categories

    @staticmethod
    def get_items_in_category(category):
        return DataHelper.category2item[category]

    @staticmethod
    def get_items_with_category_count():
        if DataHelper.item2category is None:
            return 0
        return len(DataHelper.item2category)

    @staticmethod
    def get_largest_category():
        category = max(DataHelper.category2item, key=lambda p: len(DataHelper.category2item[p]))
        return {'category': category, 'n_item': len(DataHelper.category2item[category])}

    @staticmethod
    def get_item_count():
        if DataHelper.item_ids is None:
            return 0
        return len(DataHelper.get_items())

    @staticmethod
    def get_solution_count():
        return len(DataHelper.solution_ids)

    @staticmethod
    def get_category_count():
        if DataHelper.category2item is None:
            return 0
        return len(DataHelper.category2item)

    @staticmethod
    def get_entity_id(entity):
        return DataHelper.df_entity2id[DataHelper.df_entity2id['entity'].isin(entity)]['id'].astype(int).tolist()

    @staticmethod
    def get_items(copy=False):
        if copy:
            return DataHelper.item_ids[:]
        return DataHelper.item_ids

    @staticmethod
    def get_entity_count():
        if DataHelper.df_entity2id is None:
            return 0
        return len(DataHelper.df_entity2id)

    @staticmethod
    def get_relation_count():
        if DataHelper.df_relation2id is None:
            return 0
        return len(DataHelper.df_relation2id)

    @staticmethod
    def get_relation_id(relation):
        return DataHelper.df_relation2id[DataHelper.df_relation2id['relation'] == relation]['id'].iloc[0]

    @staticmethod
    def set_data(data):
        DataHelper.data = data

        adjacency = {'h': set(), 't': set(), 'r': {}}
        for r in DataHelper.df_relation2id['id']:
            adjacency['r'][r] = {'h': set(), 't': set(), 'h_map': {}, 't_map': {}}
            r_triples = DataHelper.data.loc[DataHelper.data['r'] == r, ['h', 't']]
            for triple in r_triples.itertuples():
                _, h, t = triple
                adjacency['h'].add(h)
                adjacency['t'].add(t)
                adjacency['r'][r]['h'].add(h)
                adjacency['r'][r]['t'].add(t)
                adjacency['r'][r]['h_map'].setdefault(h, set()).add(t)
                adjacency['r'][r]['t_map'].setdefault(t, set()).add(h)
        DataHelper.adjacency = adjacency

    @staticmethod
    def get_data():
        return DataHelper.df_triple2id

    @staticmethod
    def get_solutions():
        return DataHelper.data[DataHelper.data['r'] == 0]

    @staticmethod
    def sol2triple(solutions):
        temp = []
        for row in solutions.itertuples():
            _, h, t = row
            temp += [[h, int(i), 0] for i in t.split('\t')]
        triples = pd.DataFrame(temp, columns=['h', 't', 'r'])
        return triples

    @staticmethod
    def sol2mat(solutions):
        mat = np.zeros((len(solutions), DataHelper.get_item_count()))
        rows, cols = [], []
        for row in solutions.itertuples():
            index, h, t = row
            if t:
                temp = [int(i) for i in t.split('\t')]
                rows += [index] * len(temp)
                cols += temp
        mat[rows, cols] = 1
        return mat

    @staticmethod
    def get_item_distance(item_id):
        return DataHelper.item_distance[item_id]

    @staticmethod
    def corrupt_tail(triple, n=1):
        _, h, t, r = triple
        corrupt_tails = DataHelper.adjacency['r'][r]['t'].difference(DataHelper.adjacency['r'][r]['h_map'][h])
        t_corrupt = []

        if r == 0:
            # Sampling items by distance
            if DataHelper.sampling == Sampling.DISTANCE or DataHelper.sampling == Sampling.DISTANCE_IN_CATEGORY:
                distances = DataHelper.get_item_distance(t)

                if DataHelper.sampling == Sampling.DISTANCE_IN_CATEGORY:
                    # For items with known category, sample items within the same category
                    if DataHelper.has_category(t):
                        corrupt_tails = corrupt_tails.intersection(set(DataHelper.get_items_in_category(DataHelper.get_category(t))))

                distances = distances[list(corrupt_tails)]
                distances /= distances.sum()

                t_corrupt = np.random.choice(range(len(distances)), n, p=distances).tolist()

            # Sampling items uniformly
            elif DataHelper.sampling == Sampling.UNIFORM:
                t_corrupt = random.sample(corrupt_tails, min(n, len(corrupt_tails)))
        else:
            h_primes = DataHelper.adjacency['r'][r]['h'].difference({h})

            if not len(h_primes) or not len(corrupt_tails):
                t_corrupt = random.sample(DataHelper.adjacency['t'].difference(DataHelper.adjacency['r'][r]['h_map'][h]), n)
            else:
                t_corrupt = random.sample(corrupt_tails, min(n, len(corrupt_tails)))

        return t_corrupt

    @staticmethod
    def get_batch(data, batch_size, negative2positive_ratio=1, category=False, partial_data=None, complete_data=None):
        batch_index = 0
        if category:
            columns = ['h', 't', 'r', 'nh', 'nt', 'nr', 't_category', 'nt_category']
        else:
            columns = ['h', 't', 'r', 'nh', 'nt', 'nr']

        while batch_index < len(data.index):
            sample = data.iloc[batch_index: batch_index+batch_size]
            triples = []
            for x in sample.itertuples():
                temp = DataHelper.corrupt_tail(x, negative2positive_ratio)
                _, h, t, r = x
                if category:
                    t_category = DataHelper.get_category_encoding(t)
                    triples += [[h, t, r, h, i, r, t_category, DataHelper.get_category_encoding(i)] for i in temp]
                else:
                    triples += [[h, t, r, h, i, r] for i in temp]
            batch = pd.DataFrame(triples, columns=columns)

            if partial_data is not None:
                partial_sample = partial_data[batch_index: batch_index + batch_size]
                complete_sample = complete_data[batch_index: batch_index + batch_size]
                yield {'triple': [list(i) for i in zip(*batch.values)], 'solution': [partial_sample] + [complete_sample]}
            else:
                yield [list(i) for i in zip(*batch.values)]
            batch_index += batch_size

    @staticmethod
    def get_batch_solution(partial_data, complete_data, batch_size):
        batch_index = 0
        while batch_index < len(partial_data):
            partial_sample = partial_data[batch_index: batch_index + batch_size]
            complete_sample = complete_data[batch_index: batch_index + batch_size]
            yield {'partial_solution': partial_sample, 'complete_solution': complete_sample}
            batch_index += batch_size

    @staticmethod
    def get_test_batch(data, batch_size):
        batch_index = 0
        while batch_index < len(data):
            yield [data[batch_index: batch_index + batch_size] * DataHelper.get_item_count(),
                   list(range(DataHelper.get_item_count())), [0] * DataHelper.get_item_count()]
            batch_index += batch_size
