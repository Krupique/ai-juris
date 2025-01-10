import pandas as pd
import numpy as np
import nltk
import shutil
import evaluate
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class AiJurisEval():
    def __init__(self):
        # Load the final tokenizer saved on the disk
        self.final_tokenizer = AutoTokenizer.from_pretrained('models/saved_model')

        # Load the final model saved on the disk
        self.model = AutoModelForSeq2SeqLM.from_pretrained('models/saved_model')


    def generate_text(self, input_text):
        tokenized_input_text = self.tokenizer_final(input_text, return_tensors="pt").input_ids

        tokenized_output_text = self.model.generate(tokenized_input_text, max_length = 100, temperature = 0.3, do_sample=True)

        output_text = self.tokenizer_final.decode(tokenized_output_text[0], skip_special_tokens=True)
