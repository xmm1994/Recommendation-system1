import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

def get_user_items(df):
    user_items = defaultdict()
    for user, group in df[['user_id', 'item_id']].groupby(['user_id']):
        user_items[user] = set(group['item_id'].values)

    return user_items


def get_item_users(df):
    item_users = defaultdict()
    for item, group in df[['user_id', 'item_id']].groupby(['item_id']):
        item_users[item] = set(group['user_id'].values)

    return item_users


def data_reader(is_user_items = True):
    df = pd.read_csv('data/movielens/ratings.dat', sep='::', names = ['user_id', 'item_id', 'rating', 'timestamp'])
    train, test = train_test_split(df, test_size=0.2, stratify=df['user_id'], random_state=0)
    if is_user_items:
        train_user_items = get_user_items(train)
        test_user_items = get_user_items(test)
        return train_user_items, test_user_items
    else:
        train_item_users = get_item_users(train)
        test_user_items = get_user_items(test)
        return train_item_users, test_user_items
