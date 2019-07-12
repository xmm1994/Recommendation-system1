import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

class PersonalRank(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_data, self.test_data = self.data_reader()
        self.user_items , self.item_users = self.init_graph()  #{user: item set}, {item: user set}

    def data_reader(self):
        df = pd.read_csv(self.data_path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'])
        train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['user_id'], random_state=0)

        return train_data, test_data

    def init_graph(self):
        user_items = defaultdict()
        for user, group in self.train_data[['user_id', 'item_id']].groupby(['user_id']):
            user_items[user] = set(group['item_id'].values)
        item_users = defaultdict()
        for item, group in self.train_data[['user_id', 'item_id']].groupby(['item_id']):
            item_users[item] = set(group['user_id'].values)

        return user_items, item_users

    def personalrank(self, user_items, item_users, alpha, user_root, epoches):
        user_rank = {x: 0 for x in user_items.keys()}
        item_rank = {x: 0 for x in item_users.keys()}
        user_rank[user_root] = 1
        for _ in epoches:
            tmp = {x: 0 for x in item_users.keys()}
            for user, items in user_rank.items():
                for item in items:
                    tmp[item] += alpha * item_rank[item] / len(items)
            item_rank = tmp

            tmp = {x: 0 for x in user_items.keys()}
            for item, users in item_rank.items():
                for user in users:
                    tmp[user] += alpha * user_rank[user] / len(users)
            tmp[user_root] += (1 - alpha)
            user_rank = tmp

        return item_rank


