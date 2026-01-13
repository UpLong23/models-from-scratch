# %%
import numpy as np 
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import utils

# %% [markdown]
# ### Useful notion
# 
# * Information gain: $IG = E(\text{parent}) - [\text{weighted average}]\cdot E(\text{children})$, where $E$ is the entropy, intuitively the higher the entropy is the higher the lack of order is and the lower it is the more order exists in our data.
# 
# * The (binary) entropy is defined as $$E(X) = -\sum_x p(X=x)\cdot \log_2(p(X=x))$$ where $p(X=x)=\frac{\#x}{n}$ where $n$ is the size of the dataset.
# 
# * Some stopping criteria: maximum depth of the tree, minimum number of samples in each category, min impurity decrease (minimum entropy change for a new split to be made)
# 
#  
# 

# %%
class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None,value=None):
        '''
        feature: feature used for splitting criterion
        threshold: splitting criterion
        left: left subtree
        right: left subtree

        value: (in case of a leaf node) the value of the node 
        '''
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

        # in case of a leaf node
        self.value = value
    
    def is_leaf_node(self):
        '''
        returns: if a Node object is leaf 
        '''
        return self.value is not None

# %%
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        '''
        Early stoppage criterions with hyperparams min_samples, max_depth
        min_samples: minimum number of samples in node to stop
        max_depth: maximum allowed tree depth

        n_features: number of features to take into account for the split

        root: root Node  
        '''
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        

    def fit(self, X, y):
        '''
        X: training dataset
        y: labels

        returns: DecisionTree object
        '''
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    
    def _grow_tree(self, X, y, depth=0):
        '''
        Recursive function that creates new substrees

        depth: counter of the depth of the tree so far
        returns: a Node object with the best feature for a split, the best threshold and the left and right subtrees
        '''
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria 
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False) # n_feats is number of features that we have and self.n_features is the number of features that we want to select

        # find the best split 
        best_thresh, best_feature = self._best_split(X, y, feat_idxs) # feat_idx creates the randomness in decision trees

        # create child nodes 
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        '''
        Arguments: X, y
        feat_idxs: the randomly chosen feature indices that we are going to find the best possible split 

        returns: 
            split_idx: the index of the split feature that provides the best INFORMATION GAIN
            split_threshold: best found threshold of the feature above
        '''
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx] # one of the columns that will be checked if provides the best split feature
            thresholds = np.unique(X_column) 
            
            '''We define thresholds to be all the possible values met in the selected feature
            this is naive since the best value for the split depends on multiple
            factors e.g. if the feature contains numerical or categorical values. If the values
            are categorical then the np.unique() is fine but if the values are numerical we may need 
            something more complex
            '''

            for thr in thresholds:
                # calculate IG
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain: # i.e. I choose the splitting to be done on the feature that provides the most insight
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_threshold, split_idx
    

    def _information_gain(self, y, X_column, thr):
        # parent entropy
        parent_entropy = self._entropy(y)
        # create children 
        left_idxs, right_idxs = self._split(X_column, thr)
        if len(left_idxs) == 0 or len(right_idxs) == 0: 
            return 0 # i.e. if the split is null then I don't have any information gain
    
        #calculate the weighted entropy of children 
        n = len(y) # total amount of data
        n_l, n_r = len(left_idxs), len(right_idxs) # relative amounts of data after the split 
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r # weighted entropy of the children after the best possible split
        
        # calculate IG
        information_gain = parent_entropy - child_entropy
        return information_gain


    def _split(self, X_column, split_thr):
        '''
        Split a data column based on some threshold
        X_column: the selected column of the dataset 
        split_thr: threshold to make the split

        returns: left_idxs, right_idxs i.e. all the indices of the column that satisfy the threshold 
                    and fall out of the threshold respectively
        '''
        left_idxs = np.argwhere(X_column<=split_thr).flatten() # we flatten because np.argwhere returns a shape(#samples, 1)
        right_idxs = np.argwhere(X_column>split_thr).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log2(p) for p in ps if p>0])

    def _most_common_label(self, y):
        '''
        returns: most common label in the remaing samples
        '''
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
        

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        


# %% [markdown]
# ### Testing

# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()
# pd.DataFrame(data=np.c_[data['data'], data['target']], columns=list(data['feature_names'])+['target'])

X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.2)

# %%
tree = DecisionTree()
tree.fit(X_train, y_train)

# %%
import sklearn.tree
from sklearn.metrics import classification_report

clf = sklearn.tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# %%
print(classification_report(y_test, tree.predict(X_test)))

# %%
print(classification_report(y_test, clf.predict(X_test)))

# %%
print(utils.accuracy(y_true=y_test, y_pred=tree.predict(X_test)))



# %%
