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
$ python dump_data_labels.py HogSizeFeature
```
The pkl file will be saved here: 
```
pkl/data_label/data_label_HogSizeFeature.pkl
```

### Prepare 2. Train data and tune parameters for the best classifier, then dump it as a pkl file
```sh
$ python create_classifier.py pkl/data_label/data_label_HogSizeFeature.pkl HogSizeFeature
```
The pkl file will be saved here (File name can be different, because it depends on the best parameter.):
```
pkl/estimator/svm_HogSizeFeature_rbf_1000_0.001.pkl
```

As standard output, you will get tuning report something like this:
```
Best parameters set found on development set:
{'kernel': 'rbf', 'C': 1000, 'gamma': 0.001}
Grid scores on development set:
0.551 (+/-0.056) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.001}
0.073 (+/-0.001) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.0001}
0.807 (+/-0.023) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.001}
0.558 (+/-0.045) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.0001}
0.852 (+/-0.037) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.001}
0.808 (+/-0.025) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.0001}
0.864 (+/-0.043) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.001}
0.853 (+/-0.036) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.0001}
0.861 (+/-0.039) for {'kernel': 'linear', 'C': 1}
0.860 (+/-0.041) for {'kernel': 'linear', 'C': 10}
0.860 (+/-0.041) for {'kernel': 'linear', 'C': 100}
0.860 (+/-0.041) for {'kernel': 'linear', 'C': 1000}
Detailed classification report:
The model is trained on the full development set.
The scores are computed on the full evaluation set.
             precision    recall  f1-score   support

        0.0       0.81      0.83      0.82        63
        1.0       0.80      0.86      0.83        59
        2.0       0.86      0.90      0.88        48
        3.0       0.69      0.65      0.67        63
        4.0       0.78      0.73      0.75        67

avg / total       0.78      0.79      0.78       300

```

### Test classifier
For example, if you want to test all heels pictures in data/test_data/heels/ ,
```sh
$ python test_classify.py data/test_data/heels/ pkl/estimator/svm_HogSizeFeature_rbf_1000_0.001.pkl 2 HogSizeFeature
```
where 2 is the label for "dress". (See config.py)

For example, if you want to just test one heels picture data/test_data/heels/heels_test_0.jpg ,
```sh
$ python test_classify.py data/test_data/heels/heels_test_0.jpg pkl/estimator/svm_HogSizeFeature_rbf_1000_0.001.pkl 2 HogSizeFeature
```
### Show Confusion Matrix
```sh
$ python show_confusion_matrix.py pkl/data_label/data_label_HogSizeFeature.pkl pkl/estimator/svm_HogSizeFeature_rbf_1000_0.001.pkl
```
You may get something like this:

```
==================================================
# Confusion matrix (vertical: actual, horizontal: prediction)
==================================================
[[58  0  0  2  1]
 [ 0 51  0  2  1]
 [ 1  0 59  1  0]
 [ 1  1  2 64  1]
 [ 0  1  0  4 50]]
```
## References

* http://scikit-learn.org/stable/auto_examples/index.html#classification