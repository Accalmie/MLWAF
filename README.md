# Classifier driven WAF

This repo is inspired from this work : http://fsecurify.com/fwaf-machine-learning-driven-web-application-firewall/

Some techniques are also inspired from : http://nbviewer.jupyter.org/github/ClickSecurity/data_hacking/blob/master/dga_detection/DGA_Domain_Detection.ipynb

The script currently uses the following features :

+ Query 3-5 grams count calculated from bad queries 3-5 grams using CountVectorizer
+ Query length
+ Query entropy

The following classifier are currently tested :

+ Random Forest Classifier
+ Gradient Boosting Classifier
+ Linear SV Classifier
+ Decision Tree Classifier

Results on 50% of the dataset

```
[+] Done loading data
[+] Vectorizing queries
[+] Done computing ngrams
[+] Computing ngrams score
[+] Adding ngrams score feature
[+] Calculating query entropy
[+] Splitting into train and test data
[+] Fitting random forest classifier
[+] Predicting
[+] Computing results for a test set of 134150 queries
[+] We got a total of :
        98.46142377935148 % of good results
        0.9988818486768543 % of poorly classified bad queries
        0.5396943719716736 % of false positives
[+] Moving on to GBC classifier for the same train-test samples
[+] Fitting gradient Boosting Classifier
[+] Predicting
[+] Computing results for a test set of 134150 queries
[+] We got a total of :
        98.3935892657473 % of good results
        1.1643682445024226 % of poorly classified bad queries
        0.44204248975027954 % of false positives
[+] Moving on to SVM Classifier for the same train-test samples
[+] Fitting SVM classifier
[+] Predicting
[+] Computing results for a test set of 134150 queries
[+] We got a total of :
        96.95490122996645 % of good results
        2.97726425642937 % of poorly classified bad queries
        0.06783451360417443 % of false positives
[+] Moving on to Decision Tree Classifier for the same train-test samples
[+] Fitting Decision Tree classifier
[+] Predicting
[+] Computing results for a test set of 134150 queries
[+] We got a total of :
        97.92247484159523 % of good results
        0.9631010063361909 % of poorly classified bad queries
        1.11442415206858 % of false positives
```
