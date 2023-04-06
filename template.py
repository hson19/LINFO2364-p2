import sys
import copy
from queue import PriorityQueue
from collections import defaultdict
import heapq


class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.pos_vertical, self.pos_cover = spade_repr_from_transaction(
            get_transactions(pos_filepath))

        self.neg_vertical, self.neg_cover = spade_repr_from_transaction(
            get_transactions(neg_filepath))

        self.k = k
        self.P = len(self.pos_transactions)
        self.N = len(self.neg_transactions)
        self.tot = self.P+self.N

    # Feel free to add parameters to this method
    def min_top_k(self):
        dic = self.BFS()

        if dic == None:
            print("Dic is None !")
            return

        new_dic = {k: v[2] for k, v in sorted(
            dic.items(), reverse=True, key=lambda item: item[1][2])}
        k = 0
        keys = list(new_dic.keys())

        iter = 0
        while (k < self.k):

            count = new_dic[keys[iter]]

            print("{} {} {} {}".format(new_format(
                keys[iter]), dic[keys[iter]][0], dic[keys[iter]][1], dic[keys[iter]][2]))

            iter += 1
            while (iter < len(keys) and count == new_dic[keys[iter]]):
                print("{} {} {} {}".format(new_format(keys[iter]), dic[keys[iter]]
                      [0], dic[keys[iter]][1], dic[keys[iter]][2]))
                iter += 1

            if iter >= len(keys):
                break

            k += 1

    def min_top_k_wracc(self):
        dic = self.BFS(Wracc=True)

        if dic == None:
            print("Dic is None !")
            return
        
        new_dic = {k: v[3] for k, v in sorted(
            dic.items(), reverse=True, key=lambda item: item[1][3])}
        
        k = 0
        keys = list(new_dic.keys())
        
        iter = 0
        while (k < self.k):

            count = new_dic[keys[iter]]

            print("{} {} {} {} ".format(new_format(
                keys[iter]), dic[keys[iter]][0], dic[keys[iter]][1], dic[keys[iter]][3]))

            iter += 1
            while (iter < len(keys) and count == new_dic[keys[iter]]):
                print("{} {} {} {}".format(new_format(
                    keys[iter]), dic[keys[iter]][0], dic[keys[iter]][1],  dic[keys[iter]][3]))
                iter += 1

            if iter >= len(keys):
                break

            k += 1

    def BFS(self, Wracc=False):
        PosDatabase = defaultdict(list)
        PosDatabase[tuple([])] = [(i, -1)
                                  for i in range(len(self.pos_transactions))]

        NegDatabase = defaultdict(list)
        NegDatabase[tuple([])] = [(i, -1)
                                  for i in range(len(self.neg_transactions))]

        PQ = PriorityQueue()
        dic = defaultdict(list)
        BestQ = PriorityQueue()
        
        self.BFS1(tuple([]), PosDatabase, NegDatabase, dic, PQ, [], BestQ, Wracc=Wracc)
        return dic

    def get_feature_matrices(self):
        return {
            'train_matrix': [],
            'test_matrix': [],
            'train_labels': [],
            'test_labels': [],
        }

    def cross_validation(self, nfolds):
        pos_fold_size = len(self.pos_transactions) // nfolds
        neg_fold_size = len(self.neg_transactions) // nfolds
        for fold in range(nfolds):
            print('fold {}'.format(fold + 1))
            pos_train_set = {i for i in range(len(
                self.pos_transactions)) if i < fold*pos_fold_size or i >= (fold+1)*pos_fold_size}
            neg_train_set = {i for i in range(len(
                self.neg_transactions)) if i < fold*neg_fold_size or i >= (fold+1)*neg_fold_size}

            self.mine_top_k()

            m = self.get_feature_matrices()
            classifier = tree.DecisionTreeClassifier(random_state=1)
            classifier.fit(m['train_matrix'], m['train_labels'])

            predicted = classifier.predict(m['test_matrix'])
            accuracy = metrics.accuracy_score(m['test_labels'], predicted)
            print(f'Accuracy: {accuracy}')

    def BFS1(self, sequence, PosDatabases, NegDatabases, dic, PQ, support, BestQ, Wracc=False):

        if sequence != tuple([]):

            SupportPos = support[0]
            SupportNeg = support[1]

            Support = support[2]
            if Wracc == True:
                stop=self.wracc_PQ(PQ, support[3], SupportPos, SupportNeg)# support[3] is wracc
                if stop:
                    if BestQ.empty():
                        
                        return dic
                    next_candidate = BestQ.get()[1]

                    key = next_candidate[0]

                    self.BFS1(key, PosDatabases, NegDatabases,
                            dic, PQ, next_candidate[1:], BestQ, Wracc=Wracc)
                    return
            else:
                if PQ.qsize() < self.k:
                    if Support not in PQ.queue:
                        PQ.put(Support)
                        
                elif Support < PQ.queue[0]:
                    return

                
                elif Support not in PQ.queue and PQ.queue[0] < Support:
                    PQ.get()
                    PQ.put(Support)
                    

            if Wracc:
                dic[sequence] = [SupportPos, SupportNeg, Support,
                                 self.get_wracc(SupportPos, SupportNeg)]
            else:
                dic[sequence] = [SupportPos, SupportNeg, Support]
            new_pos_keys = self.get_next_databases(
                sequence, PosDatabases,  self.pos_transactions, sequence)
            new_neg_keys = self.get_next_databases(
                sequence, NegDatabases,  self.neg_transactions, sequence)

            NewSequences = set(new_pos_keys).union(set(new_neg_keys))
            
        else:  # case sequence = []
            for item in self.pos_vertical.keys():
                PosDatabases[tuple([item])] = self.pos_vertical[item]

            for item in self.neg_vertical.keys():
                NegDatabases[tuple([item])] = self.neg_vertical[item]

            NewSequences = set(PosDatabases.keys()).union(
                set(NegDatabases.keys()))
            NewSequences.remove(tuple([]))
        for key in NewSequences:
            
            pos_support = self.get_support(PosDatabases[key])
            neg_support = self.get_support(NegDatabases[key])
            tot_support = pos_support+neg_support
            
            if Wracc == False:
                BestQ.put(
                    (-tot_support, [key, pos_support, neg_support, tot_support]))
            else:
                wracc = self.get_wracc(pos_support, neg_support)
                BestQ.put(
                    (-wracc, [key, pos_support, neg_support, tot_support, wracc]))
                

        if BestQ.empty():
            
            return dic
        next_candidate = BestQ.get()[1]
        
        key = next_candidate[0]

        self.BFS1(key, PosDatabases, NegDatabases,
                  dic, PQ, next_candidate[1:], BestQ, Wracc=Wracc)

        return dic
        

    def get_support(self, database):
        cover = set()
        for elem in database:
            cover.add(elem[0])
        return len(cover)



    def get_next_databases(self, key, database, transactions, sequence):
        new_keys = set()
        for tid, pos in database[key]:

            for i in range(pos+1, len(transactions[tid])):
                new_sequences = sequence + tuple([transactions[tid][i]])
                new_keys.add(new_sequences)
                database[new_sequences].append((tid, i))
        return new_keys

    def get_wracc(self, pos_support, neg_support):
        return round(((self.P*self.N)/float(self.tot**2))*(pos_support/(self.P) - neg_support/(self.N)),5)

    def wracc_PQ(self, PQ, wracc, pos_support ,neg_support):
        if PQ.qsize() < self.k:
            
            for elem in PQ.queue:
                if wracc == elem[0]:
                    return False
            
            PQ.put((wracc,[pos_support, neg_support]))
                
        elif self.get_wracc(pos_support,0) < PQ.queue[0][0] : #check if the upper bound is better than the worst in the PQ
            #print("upperd boud worked")
            return True
        else:

            for elem in PQ.queue:
                if wracc == elem[0]:
                    return False
            if PQ.queue[0][0] < wracc :
                PQ.get()
                PQ.put((wracc, [pos_support, neg_support]))
           
        return False

def get_transactions(filepath):
    transactions = []
    with open(filepath) as f:
        new_transaction = True
        for line in f:
            if line.strip():
                if new_transaction:
                    transactions.append([])
                    new_transaction = False
                element = line.split(" ")
                assert (int(element[1]) - 1 == len(transactions[-1]))
                transactions[-1].append(element[0])
            else:
                new_transaction = True
    return transactions


def spade_repr_from_transaction(transactions):
    spade_repr = defaultdict(list)
    covers = {}
    for tid, transaction in enumerate(transactions):
        for i, item in enumerate(transaction):
            try:
                covers[item].add(tid)
            except KeyError:
                covers[item] = {tid}
            try:
                spade_repr[item].append((tid, i))
            except KeyError:
                spade_repr[item] = [(tid, i)]
    # return {'repr': spade_repr, 'covers': covers}
    return spade_repr, covers


def get_set(first_dic, second_dic):
    items = set()
    if first_dic != None:
        for key in first_dic.keys():
            if first_dic[key] != [] and first_dic[key] != None:
                items.add(key)
    if second_dic != None:
        for key in second_dic.keys():
            if second_dic[key] != [] and second_dic[key] != None:
                items.add(key)
    return items


def new_format(tuple):

    new_format = '['
    for elem in tuple:
        new_format += str(elem)+','
    new_format = new_format[:-1]
    new_format += ']'

    return new_format


def test_Test():
    K = 3
    s = Spade('datasets/Test/positive.txt', 'datasets/Test/negative.txt', K)

    s.min_top_k()


def test_wracc():
    
    K = 3
    s = Spade('datasets/Test/positive.txt', 'datasets/Test/negative.txt', K)

    s.min_top_k_wracc()


def set_to_string(test_set):
    string = ' '
    for elem in test_set:
        string += str(elem) + " "
    return string


def main():
    pos_filepath = sys.argv[1]
    neg_filepath = sys.argv[2]
    K = int(sys.argv[3])
    s = Spade(pos_filepath, neg_filepath, K)
    s.min_top_k_wracc()
    #s.min_top_k()


if __name__ == '__main__':
    
    if len(sys.argv) == 4:
        main()
    else:
        test_wracc()

    # main()
