
import re
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch import optim
import pytorch_lightning as pl



class AraBart(pl.LightningModule):
    def __init__(self, lr=0.0001):
        super().__init__()
        self.lr = lr
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Jezia/AraBART-finetuned-wiki-ar")
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return output.loss, output.logits
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_size):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        decoder_attention_mask = batch['decoder_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

        return loss
    
    def validation_step(self, batch, batch_size):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        decoder_attention_mask = batch['decoder_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

        return loss

tokenizer = AutoTokenizer.from_pretrained("Jezia/AraBART-finetuned-wiki-ar")

def summarizeText(text, mymodel):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mymodel = mymodel.to(device)

    text_encoding = tokenizer(
        text,
        max_length=1000,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    text_encoding = {key: val.to(device) for key, val in text_encoding.items()}

    generated_ids = mymodel.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=1000,
        num_beams=5,
        repetition_penalty=1.0,
        length_penalty=0.8,
        early_stopping=True
    )

    preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]

    return "".join(preds)

# Segmenting long text before feeding it to the model
def clean_text(input_text):
    final_text = []
    sentences = re.split('\.|\n', input_text) # Segmenting based on end of sentence or paragraph 
    replace = r'[^\u0621-\u064A\u0660-\u0669\u06F0-\u06F90-9]'
    size = 0
    text = ""
    for sentence in sentences:
        words = nltk.word_tokenize(sentence) 
        size+=len(words)
        text+=sentence
        text+=' '
        if (size>=1000):
            out_text = re.sub(replace, " ", text)
            final_text.append(out_text)
            size = 0
            text = ""
    final_text.append(text)

    return final_text



