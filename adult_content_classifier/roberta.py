import evaluate
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
accuracy = evaluate.load("accuracy")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def tokenize_data(data:Dataset) -> Dataset:
    tokenized_data = data.copy()
    tokenized_data = tokenized_data.map(preprocess_function, batched=True)
    tokenized_data.set_format(type="torch")
    return tokenized_data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_roberta(output_path, tokenized_dataset):
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=2, id2label=id2label, label2id=label2id)
    model.to("cuda:0")

    training_args = TrainingArguments(
    output_dir=output_path,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True)

    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
    
    trainer.train()
    return trainer
    
def evaluate_model(trainer):
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")



