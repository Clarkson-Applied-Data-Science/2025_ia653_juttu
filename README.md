# Abstractive Text Summarization using BART

## Project Overview

This repository contains a Jupyter Notebook (`Bart.ipynb`) that shows how to perform **abstractive text summarization** using Facebook’s BART model. The goal is to create short and meaningful summaries of long documents without just copying parts of the original text.

Unlike extractive summarization (which selects sentences from the original), **abstractive summarization** rewrites the content in new words. This method is widely used in fields like news, business reports, and customer feedback analysis.

We use **BART** – a powerful Transformer model with a BERT-like encoder and GPT-style decoder. It was trained by corrupting text and learning to reconstruct it, making it great at generating readable summaries. In this notebook, we fine-tune BART on custom datasets (`curated_data_subset.csv` and `curation-corpus-base.csv`) that include pairs of full articles and their summaries.

![BART](https://github.com/user-attachments/assets/c1e2da6a-d926-4354-b85a-045277d471e7)

## Instructions to download the datasets

- Clone this repository

  ```git clone git@github.com:CurationCorp/curation-corpus.git && cd curation-corpus```

- Download the article titles, summaries, urls, and dates

  ```wget https://curation-datasets.s3-eu-west-1.amazonaws.com/curation-corpus-base.csv```



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


## Input/Output Examples

Below is an example of how the model summarizes a piece of text. We provide a sample input document (excerpt) and the corresponding summary output produced by the BART model:

**Input (excerpt from an article):**

> The Indian government's policy think tank, Niti Aayog, is testing waters to employ blockchain technology in education, health and agriculture, several media reports stated.  
>  
> The top government think tank is developing a proof of concept to take advantage of the new technology in key sectors, a senior government official told The Economic Times on condition of anonymity.  
>  
> The think tank along with blockchain startup Proffer, which was founded by MIT and Harvard graduates, held a blockchain hackathon from 10 November to 13 November 2017 at IIT Delhi, a report in YourStory said.

**Output (summarization):**

> India is testing blockchain applications in education, health and agriculture, among other sectors of the economy, and is working on a proof-of-concept platform, according to an anonymous senior government official. Government think tank Niti Aayog co-hosted a blockchain hackathon alongside start-up Proffer in November. Reports the same month revealed the think tank was also developing a fraud-resistant transaction platform called IndiaChain, which is expected to be linked to the national digital identification database IndiaStack.

In this example, the model has distilled the essential information from the article. It identified the key subject (India's policy think tank testing blockchain technology), the domains affected (education, health, agriculture), and notable initiatives (a hackathon and a platform called IndiaChain). The summary is a fluent paragraph that conveys the main points of the original document in a condensed form.

*(The above input excerpt is part of a news article, and the output is the abstractive summary generated by the model. The model rephrases the content in its own words, demonstrating the abstractive capability.)*

## Citation

If you use or build upon this work in your research or projects, please cite the repository as follows (IEEE style):

```
[1] Author Name, "Abstractive Summarization using BART (Version 1.0)", GitHub Repository, 2025. [Online]. Available: https://github.com/YourUsername/YourRepoName
```

Replace "Author Name" with the name of the repository author (or organization) and update the URL to the actual repository link. This citation format follows IEEE guidelines for referencing online repositories. By citing this project, you acknowledge the effort and give credit to the developers for their work.

## License and Contributions

This project is released under the **MIT License** (see the `LICENSE` file for details). This permissive license allows both academic and commercial use of the code, as long as proper attribution is given. Researchers and professionals are encouraged to use the code for their own summarization tasks or to integrate the BART summarization model into their applications.

Contributions to this repository are welcome and encouraged. If you have improvements, bug fixes, or new features (such as support for different datasets or models), you can contribute in the following ways:

- **Fork & Pull Requests:** Fork the repository, implement your changes, and submit a pull request for review. Please include clear descriptions of your changes and the reasons behind them.
- **Issue Tracker:** For any bugs, issues, or enhancement suggestions, open an issue on GitHub. We appreciate feedback that can make this project more useful to the community.
- **Academic Collaboration:** If you are using this project for research and have ideas on how to extend it (for example, adding evaluation metrics like ROUGE scores, or experimenting with other pre-trained models), feel free to reach out. Collaborative efforts and comparative studies are welcome.

By fostering open collaboration, we hope to evolve this project further. **Join us in advancing text summarization research and applications** – whether by contributing code, sharing ideas, or using the tool to solve real-world problems. Together, we can improve the way important information is distilled and consumed.

