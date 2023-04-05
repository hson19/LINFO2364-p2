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

        new_dic = {k: v[2] for k, v in sorted(
            dic.items(), reverse=True, key=lambda item: item[1][2])}
        k = 0
        keys = list(new_dic.keys())

        iter = 0
        while (k < self.k):

            count = new_dic[keys[iter]]
            
            print("{} {} {} {}".format(new_format(keys[iter]),dic[keys[iter]][0],dic[keys[iter]][1],dic[keys[iter]][2]))
            
            iter += 1
            while (iter < len(keys) and count == new_dic[keys[iter]]):
                print("{} {} {} {}".format(new_format(keys[iter]), dic[keys[iter]]
                      [0], dic[keys[iter]][1], dic[keys[iter]][2]))
                iter += 1

            if iter >= len(keys):
                break

            k += 1

    def depth(self):
        PosDatabase = [(i,-1) for i in range(len(self.pos_transactions))]
        NegDatabase = [(i,-1) for i in range(len(self.neg_transactions))]
        PQ = PriorityQueue()
        dic=defaultdict(list)

        return self.depth1(tuple([]), PosDatabase, NegDatabase, dic, PQ,[])
        

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

    def depth1(self,sequence,PosDatabase,NegDatabase,dic,PQ,support):

        
        if sequence != tuple([]):
            
            SupportPos = support[0]
            SupportNeg = support[1]
            
            Support = support[2]
            
            if PQ.qsize() < self.k:
                if Support not in PQ.queue:
                    PQ.put(Support)
            elif Support < PQ.queue[0]:

                return
            elif Support not in PQ.queue and PQ.queue[0] < Support:
                PQ.get()
                PQ.put(Support)
                #print(Support)
               
                
            dic[sequence] = [SupportPos, SupportNeg, Support]
            NewPosDatabases = self.get_next_databases(
                PosDatabase,  self.pos_transactions,sequence)
            NewNegDatabases = self.get_next_databases(
                NegDatabase,  self.neg_transactions,sequence)


            NewSequences= set(NewPosDatabases.keys()).union(set(NewNegDatabases.keys()))
        else: #case sequence = []
            NewPosDatabases = defaultdict(list) 
            for item in self.pos_vertical.keys():
                NewPosDatabases[tuple([item])]=self.pos_vertical[item]
                
            NewNegDatabases = defaultdict(list)
            for item in self.neg_vertical.keys():
                NewNegDatabases[tuple([item])] = self.neg_vertical[item]
                
            NewSequences= set(NewPosDatabases.keys()).union(set(NewNegDatabases.keys()))
            
        SupportSequence= {}
        for key in NewSequences:
            pos_support = self.get_support(NewPosDatabases[key])
            neg_support = self.get_support(NewNegDatabases[key])
            tot_support = pos_support+neg_support
            SupportSequence[key] = [pos_support,neg_support,tot_support]
            
        SortedNewSequences = sorted(NewSequences, key=lambda x: SupportSequence[x][2], reverse=True)
        for new_sequence in SortedNewSequences:
            
            self.depth1(new_sequence,NewPosDatabases[new_sequence],NewNegDatabases[new_sequence],dic,PQ,SupportSequence[new_sequence])
        
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
    
    def get_next_databases(self,database,transactions,sequence):
        NewDatabases = defaultdict(list)

        for tid,pos in database:

            for i in range(pos+1,len(transactions[tid])):
                new_sequences = sequence + tuple([transactions[tid][i]])
                NewDatabases[new_sequences].append((tid, i))
        return NewDatabases


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
    
    new_format=[]
    for elem in tuple:
        new_format.append(elem)
    return new_format

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
    """
    if len(sys.argv)==4:
        main()
    else:
        test_Test()
    """    
    main()
