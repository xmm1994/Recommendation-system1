from collections import defaultdict
import math

def recall(recommends, test):
    '''
    :param recommends: dict形式 {user: items set}
    :param test:  dict形式 {user: items set}
    '''
    n_union = 0
    n_test = 0
    for user, items in recommends.items():
        n_union += len(items & test[user])
        n_test += len(test[user])

    return n_union / n_test


def precision(recommends, test):
    '''
    :param recommends: dict形式 {user: items set}
    :param test:  dict形式 {user: items set}
    '''
    n_union = 0
    n_recommends = 0
    for user, items in recommends.items():
        n_union += len(items & test[user])
        n_recommends += len(items)

    return n_union / n_recommends


def coverage(recommends, all_items):
    '''
    :param recommends: dict形式 {user: items set}
    :param all_items: set
    '''
    recommend_items = set()
    for user, items in recommends.items():
        for item in items:
            recommend_items.add(item)

    return len(recommend_items) / len(all_items)


def popularity(recommends, user_items):
    '''
    :param recommends: dict形式 {user: items set}
    :param user_items: dict形式 {user: items set}
    '''
    item_popolarity = defaultdict(int)
    for _, items in user_items.items():
        for item in items:
            item_popolarity[item] += 1

    popularity_sum = 0.0
    n = 0
    for _, items in recommends.items():
        n += len(items)
        for item in items:
            popularity_sum += math.log(1 + item_popolarity[item])

    return popularity_sum / n





