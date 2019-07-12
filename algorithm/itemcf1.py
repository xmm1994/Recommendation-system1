import math
from collections import defaultdict

class ItemCF(object):

    def __init__(self, item_users):
        self.item_users = item_users  #user和item行为表：{user: items set}

    def item_similarity(self):
        '''计算物品间的相似度矩阵'''

        #建立物品倒排表
        user_items = defaultdict(set)
        for item, users in self.item_users.items():
            for user in users:
                user_items[user].add(item)

        #计算物品间的余弦相似度
        N = defaultdict(int)
        C = defaultdict(dict)
        for user, items in user_items.items():
            for i1 in items:
                N[i1] += 1
                for i2 in items:
                    if i1 != i2:
                        C[i1].setdefault(i2, 0)
                        C[i1][i2] += 1 / math.log(1 + len(user_items[user]))

        for item, related_items in C.items():
            for related_item, _ in related_items.items():
                C[item][related_item] = C[item][related_item] / math.sqrt(N[item] * N[related_item])

        self.C = C  #item间相似度矩阵
        self.user_items = user_items

    def recomment(self, user, K, N):
        '''推荐和该user最相似的K个users有过行为且当前用户没有过行为的items'''

        recomments = defaultdict(float)
        items = self.user_items[user]
        for item in items:
            for ritem, sim in sorted(self.C[item].items(), key = lambda x: x[1], reverse=True)[0:K]:
                if ritem not in items:
                    recomments[ritem] += sim

        return dict(sorted(recomments.items(), key = lambda x: x[1], reverse=True)[:N])

    def recomment_users(self, users, K, N):
        recomments = defaultdict()
        for user in users:
            recomments[user] = set(self.recomment(user, K, N).keys())

        return recomments


