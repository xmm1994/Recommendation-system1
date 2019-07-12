import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict

class LFM(object):
    def __init__(self, k, regularization_rate, learning_rate):
        self.k = k
        self.regularization_rate = regularization_rate
        self.learning_rate = learning_rate
        self.data_reader()

    def data_reader(self):
        df = pd.read_csv('../data/movielens/ratings.dat', sep='::',
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
        train_data, test = train_test_split(df, test_size=0.2, stratify=df['user_id'], random_state=0)
        self.all_items = list(train_data['item_id'].values)  # item list, 不去重， 流行度越大的物品出现次数越多
        self.train_data = defaultdict(dict)    #{user: {1: pos_item set, 0: neg_item set}}
        for user, group in train_data[['user_id', 'item_id']].groupby(['user_id']):
            self.train_data[user][1] = set(group['item_id'].values)
            self.train_data[user][0] = self.random_select_negative_sample(self.train_data[user][1], self.all_items)
        self.test = defaultdict()          #{user: item set}
        for user, group in test[['user_id', 'item_id']].groupby(['user_id']):
            self.test[user] = set(group['item_id'].values)

    def random_select_negative_sample(self, pos_items, all_items):
        '''随机采样每个用户的负样本'''
        neg_items = set()
        pos_len = len(pos_items)
        all_len = len(set(all_items))
        n = 0
        for _ in range(all_len * 3):
            item = all_items[random.randint(0, len(all_items) - 1)]
            if item not in pos_items:
                neg_items.add(item)
                n += 1
                if n >= pos_len:
                    break

        return neg_items

    def build_matrix(self):
        '''初始化隐语义向量'''
        self.p_u = defaultdict()
        self.q_i = defaultdict()
        for user in self.train_data.keys():
            self.p_u[user] = np.random.normal(size=(self.k))
        for item in set(self.all_items):
            self.q_i[item] = np.random.normal(size=(self.k))

    def train(self, epoches):
        for epoch in range(epoches):
            print('epoch %d loss: %f'%(epoch, self.loss()))
            for user, item_dict in self.train_data.items():
                for item in item_dict[1]:
                    pred = self.predict_user_item(user, item)
                    self.p_u[user] += self.learning_rate * (
                                      self.q_i[item] * (1 - pred) +
                                      self.regularization_rate * self.p_u[user])
                    self.q_i[item] += self.learning_rate * (
                                      self.p_u[user] * (1 - pred) +
                                      self.regularization_rate * self.q_i[item])
                for item in item_dict[0]:
                    pred = self.predict_user_item(user, item)
                    d = self.learning_rate * (self.q_i[item] * (0 - pred) +
                                              self.regularization_rate * self.p_u[user])
                    self.p_u[user] += d
                    d = self.learning_rate * (self.p_u[user] * (0 - pred) +
                                              self.regularization_rate * self.q_i[item])
                    self.q_i[item] += d

    def predict_user_item(self, user, item):
        return np.dot(self.p_u[user], self.q_i[item])

    def loss(self):
        C = 0
        for user, item_dict in self.train_data.items():
            for item in item_dict[1]:
                pred = self.predict_user_item(user, item)
                C += (1 - pred) * (1 - pred)
            for item in item_dict[0]:
                pred = self.predict_user_item(user, item)
                C += (0 - pred) * (0 - pred)
        for user in self.p_u.keys():
            C += self.regularization_rate * np.sum(np.square(self.p_u[user]))
        for item in self.q_i.keys():
            C += self.regularization_rate * np.sum(np.square(self.q_i[item]))

        return C

    def recomment(self, user, N):
        '''推荐预测评分最高且当前用户没有过行为的N个items'''

        recomments = defaultdict(float)
        pos_items = self.train_data[user][1]
        for item in self.q_i.keys():
            if item not in pos_items:
                recomments[item] = self.predict_user_item(user, item)

        return dict(sorted(recomments.items(), key = lambda x: x[1], reverse=True)[:N])

    def recomment_users(self, N):
        users = set(self.test.keys())
        recomments = defaultdict()
        for user in users:
            recomments[user] = set(self.recomment(user, N).keys())

        return recomments


if __name__ == '__main__':
    lfm = LFM(k=5, regularization_rate=0.01, learning_rate=0.02)
    print('build matrix: ')
    lfm.build_matrix()
    print('start train:')
    lfm.train(epoches=100)



