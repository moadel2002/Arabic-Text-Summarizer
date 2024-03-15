# Arabic Text Summarizer

This repository contains code for an abstractive summarization model designed specifically for Arabic text. The project focuses on generating concise and coherent summaries that capture essential information from longer documents. The model employs transformer-based architecture, specifically AraBart, showcasing its effectiveness in addressing the complexities of Arabic text.


## Results

The efficacy of our model was evaluated on the XL-Sum dataset. Our model achieved a remarkable ROUGE-L score of 27.839 on the test set of the XL-Sum dataset.

### ROUGE-L Scores of the Test Set
![rougel](https://github.com/moadel2002/Arabic-Text-Summarizer/assets/110140891/40c98b23-d19e-4378-b112-502c9144a96d)


But in abstractive summarization ROUGE-L score is not enough as a significant aspect of abstractive summarization quality lies in the semantic similarity between the generated summaries and the baseline summaries. In this regard, our model demonstrated a substantial semantic similarity score of 93.1. This high score is indicative of the close alignment between the content and context of the generated summaries and the baseline summaries.

### Semantic Similarity Scores of the Test Set
![semanticsim](https://github.com/moadel2002/Arabic-Text-Summarizer/assets/110140891/a06322c5-18f1-44e8-9080-9051e94da776)


## Running the Application

To run the application, clone the repo and execute the following command:

```
python app.py
```

### Arabic Text Summarizer App
<img src="https://github.com/moadel2002/Arabic-Text-Summarizer/assets/110140891/e7bff71b-8760-4580-92ce-97634ef00b7c" alt="Screenshot" width="400">


## Datasets and Weights

You can access the datasets used for training and evaluation, and weights obtained after training  from the following link: [Datasets and Weights](https://drive.google.com/drive/folders/12jOQ45-2Xw_gcHWpSiaCyZZXzoVoICaC?usp=drive_link)

