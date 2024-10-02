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
| en   	|              	|                    	|
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
### DE
### FR
### IT
### ES
