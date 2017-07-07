import pandas as pd
import numpy as np
import sklearn.feature_extraction
from sklearn.model_selection import train_test_split
import sklearn.ensemble
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import math
from collections import Counter

def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def load_data():
	bad_queries = pd.read_csv("badqueries.txt", sep="\n", header=None)
	label_bad = pd.DataFrame([1 for i in range(bad_queries.shape[0])])
	bad_queries = pd.concat([bad_queries, label_bad], axis=1)
	bad_queries.columns = ["query", "label"]


	good_queries = pd.read_csv("goodqueries.txt", sep="\n", header=None)
	label_good = pd.DataFrame([0 for i in range(good_queries.shape[0])])
	good_queries = pd.concat([good_queries, label_good], axis=1)
	good_queries.columns = ["query", "label"]

	data = pd.concat([bad_queries, good_queries])

	data["length"] = data["query"].str.len()


	data = data.sample(frac=1).reset_index(drop=True)
	filter_empty = (data["query"] != "")
	data = data[filter_empty]

	return data.sample(frac=0.5)   # Because computer is potato

def main():
	data = load_data()
	bad = data.loc[data["label"] == 1]
	print("[+] Done loading data")

	print("[+] Vectorizing queries")
	vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-4, max_df=1.0)
	ngrams_matrix = vectorizer.fit_transform(bad["query"].values.astype('U'))

	counts = np.log10(ngrams_matrix.sum(axis=0).getA1())
	ngrams_list = vectorizer.get_feature_names()
	print("[+] Done computing ngrams")

	print("[+] Computing ngrams score")
	data["n_grams"] = counts * vectorizer.transform(data["query"].values.astype('U')).T

	print("[+] Adding ngrams score feature")
	data = data.dropna()

	print("[+] Calculating query entropy")
	data['entropy'] = [entropy(s) for s in data['query']]

	y = data["label"]
	X = data.drop("label", axis=1) 
	X = X.drop("query", axis=1)

	X = X.values
	y = y.values

	print("[+] Splitting into train and test data")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	clf = sklearn.ensemble.RandomForestClassifier(n_estimators=20)

	print("[+] Fitting random forest classifier")
	clf.fit(X_train, y_train)

	print("[+] Predicting")
	y_pred = clf.predict(X_test)


	print("[+] Computing results for a test set of " + str(len(y_pred)) + " queries")
	y_test_list = y_test.tolist()

	length = len(y_test_list)
	n_good = 0
	n_false_positive = 0
	n_bad = 0

	for i in range(length):
		if y_pred[i] == 1 and y_test_list[i] == 1:
			n_good += 1
		if y_pred[i] == 1 and y_test_list[i] == 0:
			n_false_positive += 1
		if y_pred[i] == 0 and y_test_list[i] == 1:
			n_bad += 1
		if y_pred[i] == 0 and y_test_list[i] == 0:
			n_good += 1

	print("[+] We got a total of :")

	print("\t" + str((n_good / length)*100) +  " % of good results")
	print("\t" + str((n_bad / length)*100) + " % of poorly classified bad queries")
	print("\t" + str((n_false_positive / length)*100) + " % of false positives")

	print("[+] Moving on to GBC classifier for the same train-test samples")
	print("[+] Fitting gradient Boosting Classifier")
	gbc = sklearn.ensemble.GradientBoostingClassifier()
	gbc.fit(X_train, y_train)

	print("[+] Predicting")
	y_pred = gbc.predict(X_test)


	print("[+] Computing results for a test set of " + str(len(y_pred)) + " queries")
	y_test_list = y_test.tolist()

	length = len(y_test_list)
	n_good = 0
	n_false_positive = 0
	n_bad = 0

	for i in range(length):
		if y_pred[i] == 1 and y_test_list[i] == 1:
			n_good += 1
		if y_pred[i] == 1 and y_test_list[i] == 0:
			n_false_positive += 1
		if y_pred[i] == 0 and y_test_list[i] == 1:
			n_bad += 1
		if y_pred[i] == 0 and y_test_list[i] == 0:
			n_good += 1

	print("[+] We got a total of :")

	print("\t" + str((n_good / length)*100) +  " % of good results")
	print("\t" + str((n_bad / length)*100) + " % of poorly classified bad queries")
	print("\t" + str((n_false_positive / length)*100) + " % of false positives")

	print("[+] Moving on to SVM Classifier for the same train-test samples")

	clf = svm.LinearSVC()
	print("[+] Fitting SVM classifier")
	clf.fit(X_train, y_train)

	print("[+] Predicting")
	y_pred = clf.predict(X_test)


	print("[+] Computing results for a test set of " + str(len(y_pred)) + " queries")
	y_test_list = y_test.tolist()

	length = len(y_test_list)
	n_good = 0
	n_false_positive = 0
	n_bad = 0

	for i in range(length):
		if y_pred[i] == 1 and y_test_list[i] == 1:
			n_good += 1
		if y_pred[i] == 1 and y_test_list[i] == 0:
			n_false_positive += 1
		if y_pred[i] == 0 and y_test_list[i] == 1:
			n_bad += 1
		if y_pred[i] == 0 and y_test_list[i] == 0:
			n_good += 1

	print("[+] We got a total of :")

	print("\t" + str((n_good / length)*100) +  " % of good results")
	print("\t" + str((n_bad / length)*100) + " % of poorly classified bad queries")
	print("\t" + str((n_false_positive / length)*100) + " % of false positives")

	print("[+] Moving on to Decision Tree Classifier for the same train-test samples")

	clf = DecisionTreeClassifier(random_state=0)
	print("[+] Fitting Decision Tree classifier")
	clf.fit(X_train, y_train)

	print("[+] Predicting")
	y_pred = clf.predict(X_test)


	print("[+] Computing results for a test set of " + str(len(y_pred)) + " queries")
	y_test_list = y_test.tolist()

	length = len(y_test_list)
	n_good = 0
	n_false_positive = 0
	n_bad = 0

	for i in range(length):
		if y_pred[i] == 1 and y_test_list[i] == 1:
			n_good += 1
		if y_pred[i] == 1 and y_test_list[i] == 0:
			n_false_positive += 1
		if y_pred[i] == 0 and y_test_list[i] == 1:
			n_bad += 1
		if y_pred[i] == 0 and y_test_list[i] == 0:
			n_good += 1

	print("[+] We got a total of :")

	print("\t" + str((n_good / length)*100) +  " % of good results")
	print("\t" + str((n_bad / length)*100) + " % of poorly classified bad queries")
	print("\t" + str((n_false_positive / length)*100) + " % of false positives")


if __name__ == '__main__':
	main()
