from datasets import load_dataset

# Load Dataset
def load_dataset_from_csv(filename):
    # Load data
    dataset = load_dataset('csv', data_files=filename)

    # Splitting into training and testint with 80/20 ratio
    dataset = dataset['train'].train_test_split(test_size = 0.2)

    # return dataset format
    return dataset



# Preprocessing function
def data_preprocess(data, tokenizer, prefix):
    # Concatenate the prefix to each question in the list of questions given in data["question"]
    inputs = [prefix + doc for doc in data['question']]

    # Uses the tokenizer to convert the processed questions into tokens with a maximum lenght of 128, truncating any that are longer
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # Tokenize the responses given in data['answer] with a maximum lenght of 512, truncating any that are longer
    labels = tokenizer(text_target = data['answer'], max_length=512, truncation=True)

    # Add the tokens of response as labels in the input dictionary of the model
    model_inputs['labels'] = labels['input_ids']

    return model_inputs