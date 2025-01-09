# AI-Juris
AI Juris: An AI-powered legal assistant app leveraging fine-tuned open-source LLMs for accurate and efficient legal support

# Project Documentation: AI Juris

## Project Description
AI Juris is an artificial intelligence-based legal assistant application. It uses fine-tuned open-source large language models (LLMs) to provide accurate and efficient legal support. This project aims to democratize access to legal tools and facilitate consultation processes in various areas of law by leveraging modern AI technologies.

---

## Project Structure

1. **Base Model**: The project utilizes the pre-trained `flan-t5-base` model, a sequence-to-sequence transformation model developed by Google.

2. **Data Source**:
   - Data was collected from the StackExchange website using web scraping techniques.
   - The collected data is structured in a question-and-answer format, which is ideal for model training.

3. **Training Refinement**:
   - The `Seq2SeqTrainingArguments` package is used to configure training parameters.
   - Training is conducted with `Seq2SeqTrainer` to fine-tune the model with custom data.

4. **Evaluation**:
   - The `Rouge` metric, one of the most common metrics for text generation tasks, is used to assess model performance.

---

## Development Workflow

### 1. Data Collection
The data collection process consists of the following steps:
- **Objective**: Obtain a reliable knowledge base in question-and-answer format.
- **Tool Used**: A Python-based web scraping algorithm.
- **Data Source**: StackExchange, a platform offering discussions in Q&A format on various topics, including legal matters.

The collected data is stored in CSV format, containing only the fields:
- Question (full text)
- Answer(s) (full text)

### 2. Data Processing
The data undergoes a pipeline of cleaning and formatting:
- Removal of HTML and special characters.
- Text normalization (lowercasing, removal of extra spaces).
- Splitting into training, validation, and test sets in appropriate proportions.

### 3. Training Configuration
The training stages include:
- **Parameter Configuration**:
  - Learning rate: `3e-4`
  - Batch size: `4`
  - Epochs: `5`
  - Scheduler: Linear

- **Implementation**:
  - Use of `Seq2SeqTrainer` from the `transformers` library.

### 4. Model Evaluation
Model evaluation is based on the `Rouge` metric, which measures the similarity between the generated text and the reference text in terms of:
- Rouge-1 (unigrams)
- Rouge-2 (bigrams)
- Rouge-L (longest common subsequence)

> Note: A detailed explanation of the Rouge metric is provided in the Jupyter notebook.

---

## Code Structure

### Main Directories
```
AI_Juris/
├── data/
│   ├── raw/
│   ├── processed/
├── app/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
├── models/
├── results/
├── notebooks/
│   ├── data/
│   ├── model/
│   ├── scrapping_data/
│   ├── train_results/
├── poetry.lock
├── pyproject.toml
├── README.md
```

### Key Files
- `notebooks/main.ipynb`: Jupyter Notebook detailing model development and experimentation.
- `notebooks/web-scrapping.ipynb`: Jupyter Notebook with details on the development and experimentation of the scraping algorithm.
- `preprocess.py`: Cleans and formats collected data.
- `train.py`: Contains the code for model training configuration and execution.
- `evaluate.py`: Evaluates the model using the `Rouge` metric.

---

## Technologies Used
- **Language**: Python 3.12.3
- **Key Libraries**:
  - `transformers`: For using `flan-t5-base` and training components.
  - `datasets`: For data handling and processing.
  - `rouge_score`: For computing evaluation metrics.
  - `pandas`: For tabular data manipulation.
- **Dependency Manager**: Poetry

---

## Usage Examples

> Input prompt example: Can I move out of state with my children if I have a custody agreement in the state?

> Answer:  A custody agreement is a legally binding agreement between the parties to a custody arrangement. If you have a custody agreement in place, you can move out of state with your children. However, it's important to note that each state has its own rules regarding moving out of state. If you have a custody agreement in place, you may need to review the terms of the agreement to determine if you can move out of state.


(Under development)