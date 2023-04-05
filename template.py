import sys
import copy
from queue import PriorityQueue
from collections import defaultdict


class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.pos_vertical, self.pos_cover = spade_repr_from_transaction(
            get_transactions(pos_filepath))

        self.neg_vertical, self.neg_cover = spade_repr_from_transaction(
            get_transactions(neg_filepath))

        self.k = k

    # Feel free to add parameters to this method
    def min_top_k(self):
        dic = self.depth()

        if dic == None:
            print("Dic is None !")
            return
        print(dic)
        new_dic = {k: v[2] for k, v in sorted(
            dic.items(), reverse=True, key=lambda item: item[1][2])}
        k = 0
        keys = list(new_dic.keys())
        print(new_dic)
        iter = 0
        while (k < self.k):

            count = new_dic[keys[iter]]

            print(keys[iter] + set_to_string(dic[keys[iter]]))
            iter += 1
            while (iter < len(keys) and count == new_dic[keys[iter]]):
                print(keys[iter] + set_to_string(dic[keys[iter]]))
                iter += 1

            if iter >= len(keys):
                break

            k += 1

    def depth(self):
        PosDatabase = [(i,-1) for i in range(len(self.pos_transactions))]
        NegDatabase = [(i,-1) for i in range(len(self.neg_transactions))]
        PQ = PriorityQueue()
        dic=defaultdict(list)

        return self.depth1([], PosDatabase, NegDatabase, dic, PQ)
        

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

    def depth1(self,sequence,PosDatabase,NegDatabase,dic,PQ):

        
        if sequence != []:
            
            SupportPos = self.get_support(PosDatabase)
            SupportNeg = self.get_support(NegDatabase)
            
            Support = SupportPos+SupportNeg
            
            if PQ.qsize() < self.k:
                if Support not in PQ.queue:
                    PQ.put(Support)
            elif Support < PQ.queue[0]:

                return
            elif Support not in PQ.queue and PQ.queue[0] < Support:
                PQ.get()
                PQ.put(Support)
                print(Support)
                
            dic[str(sequence)] = [SupportPos, SupportNeg, Support]
        

        
            NewItems= self.get_next_items(PosDatabase,self.pos_transactions).union(self.get_next_items(NegDatabase,self.neg_transactions))
        else:
            NewItems = set()
            for item in self.pos_vertical.keys():
                NewItems.add(item)
            for item in self.neg_vertical.keys():
                NewItems.add(item)
            
        for item in NewItems:
            new_sequence = sequence + [item]
            
            NewPosDatabase = self.project(PosDatabase,item,self.pos_vertical)
            NewNegDatabase = self.project(NegDatabase,item,self.neg_vertical)
            
            self.depth1(new_sequence,NewPosDatabase,NewNegDatabase,dic,PQ)
        
        return dic
        
    def get_support(self,database):
        cover=set()
        for elem in database:
            cover.add(elem[0])
        return len(cover)
    
    def project(self,database,ItemToProject,vertical):
        new_database = []
        i,j=0,0
        if ItemToProject not in vertical.keys():
            return new_database
        while( i<len(database) and j < len(vertical[ItemToProject]) ):
            if(database[i][0] < vertical[ItemToProject][j][0]):
                i+=1
            elif(database[i][0] == vertical[ItemToProject][j][0]):
                if(database[i][1] < vertical[ItemToProject][j][1]):
                    new_database.append(vertical[ItemToProject][j])
                j+=1
            else:
                j+=1

        return new_database
    
    def get_next_items(self,database,transactions):
        new_items = set()
        for tid,pos in database:
            for item in transactions[tid][pos+1:]:
                new_items.add(item)
        return new_items

""""
def depth(sequence, dic, PQ, pos_vertical, neg_vertical, pos_support, neg_support, k):

    support = pos_support+neg_support

    dic[str(sequence)] = [pos_support, neg_support, support]
    items = get_set(pos_vertical, neg_vertical)

    for item in items:
        # print("test")
        new_pos_support = get_support(item, pos_vertical)
        new_neg_support = get_support(item, neg_vertical)

        new_support = pos_support+neg_support
        if PQ.qsize() == k:  # is the PQ full ?
            current_lowest = PQ.queue[0]

            if current_lowest > new_support:

                return
            else:
                if new_support not in PQ.queue:
                    PQ.get()
                    PQ.put(new_support)
        else:
            if new_support not in PQ.queue:
                PQ.put(new_support)

        new_pos_vertical = get_new_vertical(
            item, pos_vertical)

        new_neg_vertical = get_new_vertical(
            item, neg_vertical)

        new_sequence = sequence + [item]

        depth(new_sequence, dic, PQ, new_pos_vertical,
              new_neg_vertical, new_pos_support, new_neg_support, k)

    return dic


    

def get_new_vertical(item, vertical):
    new_vertical = defaultdict(list)
    if vertical == None:
        return None
    if vertical.get(item) == None:  # if nothing for item than we can return None
        return {}

    occurence = vertical[item]
    for key in vertical.keys():  # pour chaque clé du tableau vertical

        i, j = 0, 0
        while (i < len(occurence) and j < len(vertical[key])):

            # delete
            while (j < len(vertical[key]) and vertical[key][j][0] < occurence[i][0]):
                j += 1
            if j >= len(vertical[key]):
                break

            # same transaction
            while (j < len(vertical[key]) and occurence[i][0] == vertical[key][j][0]):
                if occurence[i][1] < vertical[key][j][1]:
                    # copy.deepcopy(vertical[key][j])
                    new_vertical[key].append(vertical[key][j])
                j += 1
            if j >= len(vertical[key]):
                break

            new_j = j
            # find first element not in the same transaction
            while (new_j < len(vertical[key]) and vertical[key][new_j][0] == occurence[i][0]):
                new_j += 1
            j = new_j

            new_i = i+1
            # find first element not in the same transaction
            while (new_i < len(occurence) and occurence[new_i][0] == occurence[i][0]):
                new_i += 1
            i = new_i
    return new_vertical


def get_new_cover(item, vertical):
    cover = set()
    if vertical == None:
        return None
    if item not in vertical.keys():
        return set()
    for tuple in vertical[item]:
        cover.add(tuple[0])
    return cover


def get_size(cover):
    if cover == None:
        return 0
    return len(cover)


def get_support(item, vertical):
    cover = get_new_cover(item, vertical)
    return get_size(cover)

"""
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


def test_Test():
    K = 3
    s = Spade('datasets/Test/positive.txt', 'datasets/Test/negative.txt', K)
    
    s.min_top_k()


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
    s.min_top_k()


if __name__ == '__main__':
    
    if len(sys.argv)==4:
        main()
    else:
        test_Test()
    #main()
