import sys
import pandas as pd
import numpy as np
import nltk
import shutil
import evaluate
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from preprocessing import data_preprocess, load_dataset_from_csv

class AiJuris():

    def __init__(self, filename):
        
        path = f'data/{filename}'
        print(path) 
        dataset = load_dataset_from_csv(path)
        
        # Load Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')

        # Load pretrained LLM
        self.model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

        # Data collator to concatenate the tokenizer and the model
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        # Every input will receive the prefix: "answer the question"
        self.prefix = "answer the question: "

        # Applies the preprocessing function to the dataset, generating the tokenized dataset 
        self.dataset_tokenized = dataset.map(data_preprocess, batched=True, fn_kwargs={"prefix": self.prefix, "tokenizer": self.data_collator.tokenizer})

        # The "punkt" package is specifically for the task of tokenization, which involves splitting a text
        # into a list of sentences
        nltk.download("punkt", quiet = True)
        nltk.download('punkt_tab', quiet=True)

        # Defining the metric
        self.metric = evaluate.load('rouge')


    def calculate_metric(self, eval_preds):
        # Unpack the predictions and labels from the eval_preds argument
        predictions, labels = eval_preds

        # Replace all non--100 values ​​in labels with the padding token ID
        labels = np.where(labels != -100,
                        labels,
                        self.tokenizer.pad_token_id)
        
        # Decode predictions to text, ignoring special tokens
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Decode labels to text, ignoring special tokens
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Add a new line after each sentence to the decoded predictions, preparing them for ROUGE evaluation
        decoded_predictions = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_predictions]
        
        # Add a new line after each label to the decoded predictions, preparing them for ROUGE evaluation
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]


        # Calculate the ROUGE metric between predictions and decoded labels, using a stemmer
        result = self.metric.compute(predictions = decoded_predictions,
                                references = decoded_labels,
                                use_stemmer = True)
        
        # Returns the result of ROUGE metric
        return result
    
    def create_trainer(self):
        # Define the train arguments
        self.training_args = Seq2SeqTrainingArguments(output_dir = "train_results",
                                                      evaluation_strategy = "epoch",
                                                      learning_rate = 3e-4,
                                                      logging_dir = "logs_treino",
                                                      logging_steps = 1,
                                                      per_device_train_batch_size = 4,
                                                      per_device_eval_batch_size = 2,
                                                      weight_decay = 0.01,
                                                      save_total_limit = 1,
                                                      num_train_epochs = 1,
                                                      predict_with_generate = True,
                                                      push_to_hub = False)
        
        # Defining the trainer
        self.trainer = Seq2SeqTrainer(model = self.model,
                                args = self.training_args,
                                train_dataset = self.dataset_tokenized["train"],
                                eval_dataset = self.dataset_tokenized["test"],
                                tokenizer = self.tokenizer,
                                data_collator = self.data_collator,
                                compute_metrics = self.calculate_metric)
        

    def train(self):
        self.trainer.train()

        self.trainer.save_model("models/saved_model")


if __name__ == "__main__":

     # Load dataset
    if len(sys.argv) < 2:
        print("Type the filename")

    else:            
        filename = sys.argv[1]
        aijuris = AiJuris(filename)

        aijuris.create_trainer()
        aijuris.train()




