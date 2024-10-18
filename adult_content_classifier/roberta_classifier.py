
from typing import List

import numpy as np

# import torch
# Importing data functions from data.py
# from data_processor import load_text_data
from data_processor_roberta import load_text_data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
)


class TextClassifier:
    def __init__(self, model_name: str, input_dir: str, output_dir: str, languages: List[str]):
        """
        Initializes the TextClassifier with a model, tokenizer, and other relevant parameters.
        """
        self.model_name = model_name
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.languages = languages
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)


    def tokenize_data(self, dataset):
        """
        Tokenizes the input dataset using the XLM-Roberta tokenizer.
        """
        print("Tokenizing data...")
        return dataset.map(lambda examples: self.tokenizer(examples['text'], truncation=True, padding=True, max_length=512), batched=True)

    @staticmethod
    def compute_metrics(eval_pred):
        """
        Computes metrics (accuracy, precision, recall, f1) for evaluation.
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(self, train_dataset, val_dataset):
        """
        Trains the model using the Hugging Face Trainer API.
        """
        print("Starting training...")
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        # if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        #     self.model.to("cuda:0")
        # else:
        #     print("CUDA device not available.")
        # self.model.to("cuda:0")


        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_dir="/raid/fhgiais/opengptx/qasid/toxicity_classification/logs",
            logging_steps=10,
            load_best_model_at_end=True,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def save(self):
        """
        Saves the model and tokenizer to the output directory.
        """
        print(f"Saving model and tokenizer to {self.output_dir}...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def run(self):
        """
        Runs the entire pipeline: loading data, tokenizing, training, and saving the model.
        """
        # Load and prepare the data
        train_dataset, val_dataset = load_text_data(self.input_dir, self.output_dir, self.languages)

        # Tokenize the datasets
        train_dataset = self.tokenize_data(train_dataset)
        val_dataset = self.tokenize_data(val_dataset)

        # Train the model
        self.train(train_dataset, val_dataset)

        # Save the trained model and tokenizer
        self.save()
