# Multi Classification for images with scikit-learn
Multi classification script for product images(example: five shoes categories)

## Requirements
* opencv(>= 2.4.8)
* numpy(1.8.2)
* scipy(0.13.3)
* scikit-learn (0.16.1)
* scikit-image (0.11.3)

## Usage
### Prepare 0. extract example dataset
Extract example dataset examples.tar.gz under data/ directory.
Where examples.tar.gz (7.5MB) consists of 300 pictures pro each categories: 
* ballerinas
* boots
* heels
* sandals
* sneakers

```sh
$ tar -xvzf example_data/examples.tar.gz -C data/
```
Please check the setting of CATEGORY_CLASS_DICT in config.py.


### Prepare 1. dump feature data and labels as pkl file
```sh
$ python dump_data_labels.py
```
The pkl file will be saved here: 
```
pkl/data_label_HogSizeFeature.pkl
```

### Prepare 2. Train data and tune parameters for the best classifier, then dump it as a pkl file
```sh
$ python create_classifier.py
```
The pkl file will be saved here:
```
pkl/estimator_HogSizeFeature.pkl
```

As standard output, you will get tuning report something like this:
```
==================================================
# Tuning hyper-parameters for F1-score
==================================================
Best parameters set found on development set:
{'kernel': 'rbf', 'C': 10, 'gamma': 0.001}
Grid scores on development set:
0.853 (+/-0.019) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.001}
0.742 (+/-0.023) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.0001}
0.868 (+/-0.037) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.001}
0.848 (+/-0.022) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.0001}
0.868 (+/-0.037) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.001}
0.847 (+/-0.030) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.0001}
0.868 (+/-0.037) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.001}
0.847 (+/-0.030) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.0001}
0.837 (+/-0.023) for {'kernel': 'linear', 'C': 1}
0.837 (+/-0.023) for {'kernel': 'linear', 'C': 10}
0.837 (+/-0.023) for {'kernel': 'linear', 'C': 100}
0.837 (+/-0.023) for {'kernel': 'linear', 'C': 1000}
Detailed classification report:
The model is trained on the full development set.
The scores are computed on the full evaluation set.
             precision    recall  f1-score   support

        0.0       0.88      0.84      0.86        63
        1.0       0.90      0.80      0.85        59
        2.0       0.88      0.96      0.92        48
        3.0       0.79      0.79      0.79        63
        4.0       0.78      0.85      0.81        67

avg / total       0.85      0.84      0.84       300

```

### Test classifier
For example, if you want to test all heels pictures in data/test_data/heels/ ,
```sh
$ python test_classify.py data/test_data/heels/ 2
```
where 2 is the label for "heels". (See config.py)

For example, if you want to just test one heels picture data/test_data/heels/heels_test_0.jpg ,
```sh
$ python test_classify.py data/test_data/heels/heels_test_0.jpg 2
```
### Show Confusion Matrix
```sh
$ python show_confusion_matrix.py
```
You may get something like this:

```
==================================================
# Confusion matrix (vertical: actual, horizontal: prediction)
==================================================
[[60  0  1  2  0]
 [ 0 71  0  1  0]
 [ 0  0 59  0  0]
 [ 0  1  1 47  0]
 [ 0  1  0  1 55]]

```
## References

* http://scikit-learn.org/stable/auto_examples/index.html#classification