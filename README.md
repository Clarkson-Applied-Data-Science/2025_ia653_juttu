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

## Abstractive Summarization:

1. **Defining a Custom Dataset and DataModule for BART Fine-Tuning**:
   ```python
   import pytorch_lightning as pl

   class Dataset(torch.utils.data.Dataset):
    """Class used as a dataset loader with defined overridden methods as required by
    #pytorch DataLoader.
    
    For more information about Dataset, Dataloader read:
    
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, texts, summaries, tokenizer, source_len, summ_len):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.source_len  = source_len
        self.summ_len = summ_len

    def __len__(self):
        return len(self.summaries) - 1

    def __getitem__(self, index):
        text = ' '.join(str(self.texts[index]).split())
        summary = ' '.join(str(self.summaries[index]).split())

        # Article text pre-processing
        source = self.tokenizer.batch_encode_plus([text],
                                                  max_length=self.source_len,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt')
        # Summary target pre-processing
        target = self.tokenizer.batch_encode_plus([summary],
                                                  max_length=self.summ_len,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt')

        return (
            source['input_ids'].squeeze(), 
            source['attention_mask'].squeeze(), 
            target['input_ids'].squeeze(),
            target['attention_mask'].squeeze()
        )

   class BARTDataLoader(pl.LightningDataModule):
    #Pytorch Lightning DataModule for BART fine-tuning.
    
    def __init__(self, tokenizer, text_len, summarized_len, file_path,
                 corpus_size, columns_name, train_split_size, batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.summarized_len = summarized_len
        self.input_text_length = summarized_len
        self.file_path = file_path
        self.nrows = corpus_size
        self.columns = columns_name
        self.train_split_size = train_split_size
        self.batch_size = batch_size

    def prepare_data(self):
        data = pd.read_csv(self.file_path, nrows=self.nrows, encoding='latin-1')
        data = data[self.columns]
        data.iloc[:, 1] = 'summarize: ' + data.iloc[:, 1]
        self.text = list(data.iloc[:, 0].values)
        self.summary = list(data.iloc[:, 1].values)

    def setup(self, stage=None):
        X_train, y_train, X_val, y_val = train_test_split(
            self.text, self.summary
        )
        self.train_dataset = (X_train, y_train) 
        self.val_dataset = (X_val, y_val)

    def train_dataloader(self):
        train_data = Dataset(texts=self.train_dataset[0],
                             summaries=self.train_dataset[1],
                             tokenizer=self.tokenizer,
                             source_len=self.text_len,
                             summ_len=self.summarized_len)
        return DataLoader(train_data, self.batch_size)

    def val_dataloader(self):
        val_dataset = Dataset(texts=self.val_dataset[0],
                              summaries=self.val_dataset[1],
                              tokenizer=self.tokenizer,
                              source_len=self.text_len,
                              summ_len=self.summarized_len)
        return DataLoader(val_dataset, self.batch_size)
   ```
  - After this cell, we have defined the data handling components: a dataset class to tokenize data on-the-fly, and a LightningDataModule (BARTDataLoader) that will manage reading a CSV and providing train/val DataLoaders.

  - Output: This cell defines classes and has no direct output if everything is correct. (No print statements here.) If something were wrong (like missing imports), we’d see errors, but assuming all is well, it runs silently.

2. **Defining the LightningModule for BART Fine-Tuning**:
  ```python
  class AbstractiveSummarizationBARTFineTuning(pl.LightningModule):
    """Abstractive summarization model class"""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.training_losses = []
        self.validation_losses = []

    def forward(self, input_ids, attention_mask, decoder_input_ids,
                decoder_attention_mask=None, lm_labels=None):
        """Model forward pass"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=decoder_input_ids  # You’re using labels=decoder_input_ids
        )
        return outputs

    def preprocess_batch(self, batch):
        """Reformatting batch"""
        input_ids, source_attention_mask, decoder_input_ids, decoder_attention_mask = batch
        y = decoder_input_ids
        return input_ids, source_attention_mask, decoder_input_ids, decoder_attention_mask, y

    def training_step(self, batch, batch_idx):
        input_ids, source_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels = self.preprocess_batch(batch)
        outputs = self.forward(input_ids, source_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels)
        loss = outputs.loss
        self.training_losses.append(loss.detach())  # Store for epoch summary
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, source_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels = self.preprocess_batch(batch)
        outputs = self.forward(input_ids, source_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels)
        loss = outputs.loss
        self.validation_losses.append(loss.detach())  # Store for epoch summary
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_losses).mean()
        self.log('epoch', self.current_epoch)
        self.log('avg_epoch_loss', avg_loss, prog_bar=True)
        self.training_losses.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_losses).mean()
        self.log('val_avg_epoch_loss', avg_loss, prog_bar=True)
        self.validation_losses.clear()

    def configure_optimizers(self):
        return AdamW(self.model.parameters())
  ```
  - Output: Defining the class produces no immediate output. If there were mistakes in the code, we’d see errors. Otherwise, it’s silent.

  - Now we have a LightningModule that knows how to take our tokenized batches, feed them to BART, compute the loss, and how to optimize the model.


3. **Initializing the Pre-trained BART Model and DataModule**:
   ```python
   # Tokenizer
   # Upload the curated_data_subset.csv if using colab or change the path to local file
   from transformers import BartForConditionalGeneration, BartTokenizer

   model_ = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
   tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

   # Dataloader
   dataloader = BARTDataLoader(tokenizer=tokenizer, text_len=512,
                            summarized_len=150,
                            file_path='curated_data_subset.csv',
                            corpus_size=50, columns_name=['article_content','summary'],
                            train_split_size=0.8, batch_size=2)
   # Read and pre-process data
   dataloader.prepare_data()

   # Train-test Split
   dataloader.setup()
   ```
   - Output: There is no direct print output in this cell. However, behind the scenes, from_pretrained might output something:

      - It could print a message about downloading the model if it’s not cached. If already cached, it might be silent.

      - In our recorded run, there’s no explicit output shown for these lines, which suggests the model and tokenizer loaded without issues or verbose messages.

      - The DataLoader methods don’t print anything either. So this cell likely produces no user-visible output.

4. **Setting up the Trainer and Starting Fine-Tuning**:
   ```python
   from torch.utils.data import DataLoader

   trainer = pl.Trainer(
      check_val_every_n_epoch=1,
      max_epochs=5,
      accelerator="gpu",
      devices=1
   )

   # Fit model
   trainer.fit(model, dataloader)
   ```
  - Sets up a PyTorch Lightning Trainer and begins fine-tuning the model on the small dataset for 5 epochs.

5. **Defining a Summarization Function Using a Pre-trained Model**:
   ```python
   import torch
   from transformers import BartTokenizer, BartForConditionalGeneration

   def summarize_article(article, model_name='facebook/bart-large-cnn', max_input_len=1024, max_output_len=150):
    # Load model and tokenizer only once if you use this function repeatedly
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tokenize and encode the article
    inputs = tokenizer.encode(article, return_tensors='pt', max_length=max_input_len, truncation=True).to(device)

    # Generate summary
    summary_ids = model.generate(
        inputs,
        num_beams=4,
        max_length=max_output_len,
        early_stopping=True
    )

    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

   # Example usage
   article = """
    My friends are cool but they eat too many carbs.
   """

   summary = summarize_article(article)
   print("Summary:")
   print(summary)
   ```
   - This is the summary produced by the model for the toy article. It’s somewhat repetitive and not very practical (the input sentence itself was a bit trivial). The model tried to make a coherent paragraph out of it, somewhat humorously repeating the idea.
   - This shows the model can generate text; however, with such a simple input, it essentially paraphrased and added content. (The BART-large model was trained on news, so given a single statement it created a pseudo-explanatory paragraph that sounds like an opinion.)
   - The key point is: the function works and we got a “summary” (albeit not a shorter one in this case, because the input was so short already).

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


