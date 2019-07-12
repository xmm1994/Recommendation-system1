import math
from collections import defaultdict

class UserCF(object):

    def __init__(self, user_items):
        self.user_items = user_items  #user和item行为表：{user: items set}

    def user_similarity(self):
        '''计算用户间的相似度矩阵'''

        #建立用户倒排表
        item_users = defaultdict(set)
        for user, items in self.user_items.items():
            for item in items:
                item_users[item].add(user)

        #计算用户间的余弦相似度
        N = defaultdict(int)
        C = defaultdict(dict)
        for item, users in item_users.items():
            for u1 in users:
                N[u1] += 1
                for u2 in users:
                    if u1 != u2:
                        C[u1].setdefault(u2, 0)
                        C[u1][u2] += 1

        for u, related_users in C.items():
            for related_user, _ in related_users.items():
                C[u][related_user] = C[u][related_user] / math.sqrt(N[u] * N[related_user])

        self.C = C  #user间相似度矩阵

    def recomment(self, user, K, N):
        '''推荐和该user最相似的K个users有过行为且当前用户没有过行为的items'''

        recomments = defaultdict(float)
        items = self.user_items[user]
        for u, sim in sorted(self.C[user].items(), key = lambda x: x[1], reverse=True)[0:K]:
            for item in self.user_items[u]:
                if item not in items:
                    recomments[item] += sim

        return dict(sorted(recomments.items(), key = lambda x: x[1], reverse=True)[:N])

    def recomment_users(self, users, K, N):
        recomments = defaultdict()
        for user in users:
            recomments[user] = set(self.recomment(user, K, N).keys())

        return recomments


