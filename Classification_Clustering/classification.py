import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Classification:
    def __init__(self):
        "Implementing Different classification Algorithms"
        print("Starting with Classification")

        print("Loading data")
        iris = load_iris()
        self.X, self.y = np.array(iris.data), np.array(iris.target) #Already cleaned data ready to be processed and targets ready to be plotted
        #print(f"Training array: {self.X}, with Shape: {self.X.shape}, target array: {self.y}, with Shape: {self.y.shape}")
                
        #Plotting the graph
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y) #Because of size mismatch, we do it like this
        plt.xlabel("Sepal Length")
        plt.ylabel("Sepal Width")
        plt.show()
        print("Now lets predict")
        preds = self.knn()
        print("Alrights, lets go for decision trees")


    def distance(self, p1, p2):
        "Euclidean distance betweent the points"
        try:
            return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))
        except Exception as e:
            print(f"Error while calculating distance: {str(e)}")

    def knn(self):
        "K Nearest Neighbor Classification algorithm"
        try:
             
            k = 4 #choosing the number of neighbors
            predictions = []
            self.X_test = np.array([[5.5, 2.1, 3.3, 9.5], 
                                    [2.3, 5.7, 1.8, 6.4], 
                                    [0.5, 3.2, 6.9, 2.8],
                                    [4.4, 1.7, 5.9, 0.8],
                                    [6.2, 2.5, 1.1, 3.6],
                                    [3.9, 0.4, 6.5, 2.1],
                                    [5.6, 4.2, 0.9, 1.7],
                                    [1.3, 6.8, 2.7, 5.5],
                                    [2.8, 1.1, 4.6, 6.2],
                                    [0.9, 3.7, 5.1, 2.4],
                                    [6.5, 2.9, 3.3, 1.2]]) #Test features to be predicted
            abs_distance = []
            for points in range(len(self.X_test)):

                for i in range(len(self.X)):
                    dist = self.distance(self.X[i], points)
                    abs_distance.append((dist, self.y[i]))#takes only 1 argument that is why double braces

                abs_distance.sort(key=lambda x: x[0])

                nearest_labels = [label for _, label in abs_distance[:k]] #returns the absolute array, _ is to ignore first element

                prediction_label = Counter(nearest_labels).most_common(1)[0][0]
                predictions.append(prediction_label)

            for i, pred in enumerate(nearest_labels):
                print(f"Label for the value {self.X_test[i]} is as follows:{pred}")

            return predictions
        
        except Exception as e:
            print(f"Error received during knn: {str(e)}")

class DecisionTrees:
    def __init__(self):
        "Decision Trees class"
        print("Starting with Tress")

        print("Loading data")
        iris = load_iris()
        self.X, self.y = np.array(iris.data), np.array(iris.target) #Already cleaned data ready to be processed and targets ready to be plotted
        #print(f"Training array: {self.X}, with Shape: {self.X.shape}, target array: {self.y}, with Shape: {self.y.shape}")
        self.X_train, self.y_train = self.X[:120], self.y[:120]
        self.X_test, self.y_test = self.X[120: ], self.y[120:]
        y_act, y_pred = self.decision_trees()
        self.eval(y_act, y_pred)

    def gini(self, y): #Measures the probability of a random variable wrongly classified in a randomly chosen array
        "Gini Index for the tree"
        counts = np.bincount(y)# Occurance of each value in a non=negative array
        probs = counts / len(y)
        return 1 - np.sum((probs **2))

    def split_data(self, X, y, feat_ind, thresh):
        "Splitting dataset into each tree if a threshold is crossed"
        left_mask = X[:, feat_ind] <= thresh
        right_mask = X[:, feat_ind] > thresh
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask] 
        

    def best_split(self, X, y):
        ""
        best_gini = 1
        best_ind, best_thresh = None, None

        n_feat = X.shape[1]

        for feat in range(n_feat):
            threshold = np.unique(X[:feat])

            for t in threshold:
                _, y_left, _, y_right = self.split_data(X, y, feat, t)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_split = (len(y_left) / len(y) * self.gini(y_left) + len(y_right) / len(y) * self.gini(y_right))

                if gini_split <best_gini:
                    best_gini = gini_split
                    best_ind = feat
                    best_thresh = t
        
        return best_ind, best_thresh
    
    def build_tree(self, X, y, max_depth = 5, depth = 0):
        "Building nodal tree for our decision tree model"
        #In case we reach max_depth, generating leaf node.
        if len(set(y)) == 1:
            return Node(value=y[0]) #if only 1 node present
        
        if depth >= max_depth or len(y) <= 1: 
            most_common = Counter(y).most_common(1)[0][0] #assigning most common class to the node
            return Node(value=most_common)
        
        #finding best split
        feature, threshold = self.best_split(X, y)
        if feature is None: 
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)
        
        #Splitting Dataset
        X_left, y_left, X_right, y_right = self.split_data(X, y, feature, threshold) #Splitting in right and left branches

        left_node = self.build_tree(X_left, y_left, max_depth, depth)
        right_node = self.build_tree(X_right, y_right, max_depth, depth)

        return Node(feature, threshold, left_node, right_node) #Returning a node for the tree
    
    def predict_one(self, node, X):
        "Predicting a node"
        while node.value is None:
            if X[node.feat] < node.thresh:
                node = node.left
            else:
                node = node.right
        
        return node.value
    
    def predict(self, tree, X):
        "Predicting the entire tree"
        return np.array([self.predict_one(tree, x) for x in X])


    def decision_trees(self):
        "Going for classification of decision trees"
        self.tree = self.build_tree(self.X_train, self.y_train, max_depth=3)

        self.y_preds = self.predict(self.tree, self.X_test)

        for i, pred in enumerate(self.y_preds):
            print(f"Test Sample: {i+1}, predicted = {pred}, True = {self.y_test[i]}")
        
        return self.y_test, self.y_preds

    def eval(self, y_act, y_pred):
        "Evaluation metrics with Accuracy, precision, Recall, F1, ROC"
        self.acc = accuracy_score(y_act, y_pred)
        self.prec = precision_score(y_act, y_pred, average='macro')
        self.rec = recall_score(y_act, y_pred, average='macro')
        self.f1 = f1_score(y_act, y_pred, average='macro')

        print(f"Accuracy: {self.acc:.3f}")
        print(f"Precision: {self.prec:.3f}")
        print(f"Recall: {self.rec:.3f}")
        print(f"F1-score: {self.f1:.3f}")


class Node:
    def __init__(self,feat = None, thresh = None, left = None, right = None, *, value = None):
        "For nodes of each tree nodes"
        self.feat = feat
        self.thresh = thresh
        self.left = left
        self.right = right
        self.value = value


class Clustering:
    def __init__(self):
        "Now making kmeans clustering"
        print("Finally moving towards Clustering")
        self.X, self.y = make_blobs(n_samples = 2000 ,n_features = 2,centers = 3,random_state = 23)

        plt.figure(figsize=(14, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c = self.y, label = "Blob Dataset")
        plt.grid(True)
        plt.show()
        self.clustering_()

    def distance(self, p1, p2):
        "Euclidean distance betweent the points"
        try:
            return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))
        except Exception as e:
            print(f"Error while calculating distance: {str(e)}")
    
    def assign_clusters(self, X, clusters, k):
        "Assigning clusters to their nearest center"
        for idx in range(X.shape[0]):
            dist = []

            curr_x = X[idx]

            for i in range(k):
                dis = self.distance(curr_x, clusters[i])
                dist.append(dis)
            curr_cluster = np.argmin(dist)
            clusters.append(curr_cluster)
        
        return clusters
    
    def update_clusters(self, clusters, k):
        "Updating clusters on new centers"
        for i in range(k):
            points = np.array(clusters[i])
            if points.shape[0] > 0:
                new_cent = points.mean(axis=0)
                clusters[i] = new_cent
        return clusters

    def predict_clusters(self, X, clusters, k):
        "Predicting clusters"
        preds = []
        for i in range(X.shape[0]):
            dist = []
            for j in range(k):
                dis = self.distance(X[i], clusters[j])
                dist.append(dis)
            
            preds.append(dist)
        return preds
    
    def clustering_(self):
        "Finalized calls"
        k = 3
        self.clusters = []
        np.random.seed(23)
        for idx in range(k):
            center = 2*(2*np.random.random((self.X.shape[1],))-1)
            self.clusters.append(center)
            print(f"Centers are as follows: {center}")
        

        plt.scatter(self.X[:,0],self.X[:,1])
        plt.grid(True)
        for i in range(len(self.clusters)):
            center = self.clusters[i]
            plt.scatter(center[0], center[1], marker = '*',c = 'red')
        plt.show()

        #Now gping for the finish
        self.clusters = self.assign_clusters(self.X, self.clusters, k)
        self.clusters = self.update_clusters(self.clusters, k)
        self.y_preds = self.predict_clusters(self.X, self.clusters, k)

        y_preds = np.array(self.y_preds)
        #plotting
        plt.scatter(self.X[:,0], self.X[:,1],c = y_preds.astype(int), cmap='viridis')
        for center in self.clusters:
            plt.scatter(center[0],center[1],marker = '^',c = 'red', s=200, edgecolors='k')
        plt.show()


if __name__ == "__main__":
    #Starting the process
    cls_ = Classification()
    DT = DecisionTrees()
    Cl = Clustering()
