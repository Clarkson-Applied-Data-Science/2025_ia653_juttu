# Abstractive Text Summarization using BART

## Project Overview


This project implements an abstractive text summarization system using the BART (Bidirectional and Auto-Regressive Transformer) model, a state-of-the-art pre-trained language model developed by Facebook AI. Abstractive summarization involves generating concise summaries that capture the key ideas of a longer text, often rephrasing content in a natural and coherent way. The goal of this project is to build a robust summarizer that can process large documents (such as news articles or reports) and produce fluent, informative summaries for users.

The BART model is fine-tuned on a large benchmark dataset (CNN/DailyMail) of news articles and corresponding summaries. By training on this extensive dataset, the system learns to identify and articulate the most important points in new documents. The resulting summarizer is capable of handling diverse topics and writing styles, making it suitable for real-world applications like news aggregation, report generation, or content curation. All aspects of the system—from text preprocessing to summary generation—are designed to ensure high quality and user-friendly output.

This repository contains a Jupyter Notebook (`Bart.ipynb`) that shows how to perform **abstractive text summarization** using Facebook’s BART model. The goal is to create short and meaningful summaries of long documents without just copying parts of the original text.

Unlike extractive summarization (which selects sentences from the original), **abstractive summarization** rewrites the content in new words. This method is widely used in fields like news, business reports, and customer feedback analysis.

We use **BART** – a powerful Transformer model with a BERT-like encoder and GPT-style decoder. It was trained by corrupting text and learning to reconstruct it, making it great at generating readable summaries. In this notebook, we fine-tune BART on custom datasets (`curated_data_subset.csv` and `curation-corpus-base.csv`) that include pairs of full articles and their summaries.


![BART](https://github.com/user-attachments/assets/c1e2da6a-d926-4354-b85a-045277d471e7)

## Instructions to download the datasets

- Clone this repository

  ```git clone git@github.com:CurationCorp/curation-corpus.git && cd curation-corpus```

- Download the article titles, summaries, urls, and dates

  ```wget https://curation-datasets.s3-eu-west-1.amazonaws.com/curation-corpus-base.csv```


### **Approaches to Abstractive Summarization:**


* **Tree Based**:
  ![Untitled](https://github.com/user-attachments/assets/3f14dc4f-0a02-43fd-8933-155e203a0705)

* **Template Based**:
  ![Untitled](https://github.com/user-attachments/assets/a879e833-64ac-42f8-beca-63f535f87092)

* **Graph Based**:
  ![Untitled](https://github.com/user-attachments/assets/9082b6e0-4a21-4992-8b34-28d6ca105a47)



## Features

- **Powerful Model:** This project uses Facebook's BART model (`facebook/bart-base`), which is one of the top models for summarizing text. The notebook shows how to fine-tune it for creating high-quality summaries.

- **Paraphrased Summaries (Abstractive):** Instead of just picking sentences from the original text, the model rewrites the content in its own words. This makes the summaries more natural and easier to read.

- **Ready-to-Use Datasets:** Two CSV datasets are included:
  - `curated_data_subset.csv` – A small sample (about 50 articles) to quickly test and train the model.
  - `curation-corpus-base.csv` – A full set of article-summary pairs for deeper training and better results.

- **Step-by-Step Notebook:** The Jupyter Notebook explains everything clearly — from loading data, setting up the model, training with PyTorch Lightning, to generating summaries.

- **Custom Data Loader & Training Loop:** It includes a custom way to load and split your data for training and testing. The training process supports GPU for faster performance.

- **Try Your Own Texts:** After training, you can use the model to summarize your own documents. You can also test the pre-trained model without any extra training.

- **Easy to Modify:** You can tweak settings like input length, summary length, or use your own dataset. The code is modular and easy to adapt for different projects.

## Installation

To set up the environment for running the notebook, please follow these steps:

1. **Prerequisites:** Ensure you have **Python 3.8+** installed. We recommend using a virtual environment (via `venv` or `conda`) to manage dependencies. You will also need **Jupyter Notebook** or **Jupyter Lab** to run the `.ipynb` file.
2. **Create Environment (Optional):** If using Anaconda or Miniconda, you can create a new environment:
   ```bash
   conda create -n bart_summarization python=3.9 -y
   conda activate bart_summarization
   ```
   *Alternatively, create and activate a Python virtual environment using `python -m venv env`.* 
3. **Install Dependencies:** Install the required Python packages. The key dependencies include Hugging Face Transformers, PyTorch, PyTorch Lightning, and Pandas. You can install these via pip:
   ```bash
   pip install transformers torch pytorch-lightning pandas scikit-learn
   ```
   This will install:
   - **Transformers:** for the BART model and tokenizer (e.g., `transformers==4.x`).
   - **PyTorch:** deep learning framework (ensure you install a version compatible with your CUDA if using a GPU).
   - **PyTorch Lightning:** for a high-level training loop structure.
   - **Pandas:** for CSV data handling.
   - **scikit-learn:** for data splitting (used for train/validation split).
4. **Jupyter:** If not already installed, also install Jupyter:
   ```bash
   pip install notebook
   ```
   or use Jupyter Lab:
   ```bash
   pip install jupyterlab
   ```
5. **Verify Installation:** After installing, you can verify by importing the libraries in a Python shell:
   ```python
   import torch, transformers, pytorch_lightning, pandas
   ```
   and ensure no errors occur. If you plan to use a GPU for training, confirm that PyTorch detects it:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
   which should output `True` if a CUDA-enabled GPU is accessible.

## Input/Output Examples

    Example 1 (News Article):

        Input:
        “The Central Bank announced a reduction in interest rates today, aiming to stimulate economic growth. Several analysts note that this move could help boost borrowing and investment. The stock market reacted positively, with major indexes rising by 2% after the announcement.”

        Summary:
        “The Central Bank cut interest rates to stimulate economic growth, a move expected to encourage borrowing and investment. The stock market responded positively, with major indexes rising 2% on the news.”

    Example 2 (Research Abstract):

        Input:
        “In a recent study, researchers developed a new machine learning algorithm to predict housing prices. The model uses advanced neural network techniques and was trained on a large real estate dataset. Initial results show the new method outperforms existing approaches in accuracy.”

        Summary:
        “Researchers introduced a new machine learning model using neural networks to predict housing prices. Trained on an extensive real estate dataset, the model outperforms existing methods in accuracy.”

These examples illustrate how the system takes an input document and produces a clear, concise summary that captures the essential information.
  
## Results

The summarization system has been evaluated on the CNN/DailyMail test set using standard metrics. The performance demonstrates the model’s strong summarization capability:

    ROUGE-1: 0.481

    ROUGE-2: 0.481

    ROUGE-L: 0.4819

    ROUGE-Lsum: 0.4819

    BLEU: 98.47

These results indicate robust performance in capturing the content of the input text (as reflected by the ROUGE scores) and generating fluent, high-precision summaries (as reflected by the BLEU score). Overall, the evaluation scores show that the BART-based system produces high-quality summaries comparable to current benchmarks, confirming its effectiveness and reliability.

## Usage

1. **Get the Files:**
   - Download or clone this project from GitHub:
     ```bash
     git clone https://github.com/Clarkson-Applied-Data-Science/2025_ia653_juttu.git]\
     cd 2025_ia653_juttu
     ```
   - Make sure the two data files (`curated_data_subset.csv` and `curation-corpus-base.csv`) are in the same folder as the notebook. They come included.

2. **Open the Notebook:**
   - Start Jupyter Notebook and open `Bart.ipynb`:
     ```bash
     jupyter notebook Bart.ipynb
     ```
   - You can also use **Jupyter Lab** or **Google Colab** (upload the notebook and CSV files if using Colab).

3. **Load the Data:**
   - Run the first few cells to load the data.
   - By default, it uses `curated_data_subset.csv` (a small sample for fast testing).
   - You can switch to `curation-corpus-base.csv` for full training (this takes more time and needs more memory).

4. **Set Up the Model:**
   - The notebook loads Facebook’s BART model (`facebook/bart-base`) using Hugging Face Transformers.
   - It prepares the text data and sets default lengths:
     - Input text: 512 tokens
     - Summary: 150 tokens

5. **Train the Model:**
   - Run the training section to fine-tune BART on the dataset.
   - It uses **PyTorch Lightning** for easy training.
   - If you have a GPU, it will use it automatically.
   - If you're using the full dataset, lower the `batch_size` or `epochs` if you run into memory issues.

6. **Generate Summaries:**
   - You can test the model on new text using a helper function like `summarize_article()`.
   - Paste in your own article or use the sample ones in the notebook.
   - The model will return a short summary.

7. **Check the Output:**
   - Look at the summary and compare it with the original.
   - You’ll see how well the model captured the main idea.

8. **Try New Settings:**
   - Want different results? Change parameters like:
     - `num_beams` (for beam search quality)
     - `max_length` (for longer/shorter summaries)
   - You can also try other models like `facebook/bart-large-cnn`.



