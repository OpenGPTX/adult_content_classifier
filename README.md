# Adult content classifier

Trains a classifier ([bag of words approach](https://medium.com/@devesh_kumar/building-a-simple-spam-classifier-using-scikit-learn-d3a84e6f3112)) to classify adult content. 
Needs two data sources: 
1. Jsonl with data filtered because it was adult content
2. Jsonl with normal data

Both should be language specific. 


## Usage
Install with poetry 
```bash
poetry install
```

And run with click 
```bash 
poetry run train --help
```