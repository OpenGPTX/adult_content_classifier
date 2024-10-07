# Findings for BoW approach

This file contains the results of training a classifier with a bag of words (bow) approach to predict adult content.

## Model and Vectorizer

The Vectorizer is:
```python 
TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        use_idf=True,
        max_features=30000,  # Limit the number of features
        min_df=5,  # Ignore terms that appear in less than 5 documents
    )

```
Model is a `MultinomialNB` without any args. We have one model per language "endefrites"

## Train Data

Size of data it was trained on:
| Lang 	| Size text df 	| Size vectorized df 	|
|------	|--------------	|--------------------	|
| en   	| 127 MB      	| 175MB                	|
| de   	| 137 MB       	| 169MB              	|
| fr   	| 147 MB       	| 197MB              	|
| it   	| 161 MB       	| 204MB              	|
| es   	| 159 MB       	| 202MB              	|


## Profiling

This table presents the speed of one classifier (it) for prediction:

| Process   	| Total Time (s)       	| % of Total 	| Rows             	| Size       	| Chars                 	|
|-----------	|----------------------	|------------	|------------------	|------------	|-----------------------	|
| Sizes     	| //                   	| //         	| 10.000           	| 21.3 MB    	| 21.312.184            	|
| Vectorize 	| 5,8883428573608      	| 99.83%     	| 1.698,27 Rows/s  	| 3,617 MB/s 	| 3.619.385 Chars/s     	|
| Predict   	| 0,009656190872192383 	| 0.17%      	| 1.035,605 Rows/s 	| 2,205 GB/s 	| 2.207.100.530 Chars/s 	|
| Total     	| 5,8979990482         	| 100%       	| 1.695,490 Rows/s 	| 3,611 MB/s 	| 3.613.460 Chars/s     	|


## Evaluation

Evaluation for each language follows

### EN
Vectorized data with shape (3600, 30000)
 F1 score: 0.535833891493637
 Accuracy: 0.615
               precision    recall  f1-score   support

            0       0.59      0.79      0.67      1800
            1       0.67      0.44      0.54      1800

     accuracy                           0.61      3600
    macro avg       0.63      0.61      0.60      3600
 weighted avg       0.63      0.61      0.60      3600

 Confusion matrix:
 [True Negative, False Positive
 False Negative True Positive]
 [[1414  386]
  [1000  800]]
 [[0.39277778 0.10722222]
  [0.27777778 0.22222222]]
 Evaluation time: 0.07191276550292969

### DE
Vectorized data with shape (3410, 30000)
 F1 score: 0.620460358056266
 Accuracy: 0.5648093841642229
               precision    recall  f1-score   support

            0       0.64      0.40      0.49      1800
            1       0.53      0.75      0.62      1610

     accuracy                           0.56      3410
    macro avg       0.58      0.57      0.56      3410
 weighted avg       0.59      0.56      0.55      3410

 Confusion matrix:
 [True Negative, False Positive
 False Negative True Positive]
 [[ 713 1087]
  [ 397 1213]]
 [[0.20909091 0.31876833]
  [0.11642229 0.35571848]]
 Evaluation time: 0.021895170211791992

### FR
 F1 score: 0.5810620960944085
 Accuracy: 0.573512585812357
               precision    recall  f1-score   support

            0       0.59      0.54      0.57      1800
            1       0.56      0.61      0.58      1696

     accuracy                           0.57      3496
    macro avg       0.57      0.57      0.57      3496
 weighted avg       0.58      0.57      0.57      3496

 Confusion matrix:
 [True Negative, False Positive
 False Negative True Positive]
 [[ 971  829]
  [ 662 1034]]
 [[0.277746   0.23712815]
  [0.18935927 0.29576659]]
 Evaluation time: 0.010744571685791016

### IT

Vectorized data with shape (3276, 30000)
 F1 score: 0.5729827742520399
 Accuracy: 0.5686813186813187
               precision    recall  f1-score   support

            0       0.63      0.51      0.56      1800
            1       0.52      0.64      0.57      1476

     accuracy                           0.57      3276
    macro avg       0.58      0.58      0.57      3276
 weighted avg       0.58      0.57      0.57      3276

 Confusion matrix:
 [True Negative, False Positive
 False Negative True Positive]
 [[915 885]
  [528 948]]
 [[0.27930403 0.27014652]
  [0.16117216 0.28937729]]
 Evaluation time: 0.016904115676879883

### ES

Vectorized data with shape (3505, 30000)
 F1 score: 0.4736472241742797
 Accuracy: 0.5726105563480742
               precision    recall  f1-score   support

            0       0.56      0.74      0.64      1800
            1       0.59      0.40      0.47      1705

     accuracy                           0.57      3505
    macro avg       0.58      0.57      0.56      3505
 weighted avg       0.58      0.57      0.56      3505

 Confusion matrix:
 [True Negative, False Positive
 False Negative True Positive]
 [[1333  467]
  [1031  674]]
 [[0.38031384 0.13323823]
  [0.29415121 0.19229672]]
 Evaluation time: 0.01746678352355957