import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class AiJurisEval():
    def __init__(self):
        # Load the final tokenizer saved on the disk
        self.final_tokenizer = AutoTokenizer.from_pretrained('models/saved_model')

        # Load the final model saved on the disk
        self.model = AutoModelForSeq2SeqLM.from_pretrained('models/saved_model')


    def generate_text(self, input_text):
        tokenized_input_text = self.final_tokenizer(input_text, return_tensors="pt").input_ids

        tokenized_output_text = self.model.generate(tokenized_input_text, max_length = 100, temperature = 0.3, do_sample=True)

        output_text = self.final_tokenizer.decode(tokenized_output_text[0], skip_special_tokens=True)

        return output_text



if __name__ == "__main__":

     # Load dataset
    if len(sys.argv) < 2:
        print("Input the prompt")

    else:
        prompt = sys.argv[1]
        aijuris =  AiJurisEval()

        output_text = aijuris.generate_text(prompt)
        print(output_text)
        
