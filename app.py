from flask import Flask, render_template, request
from model import AraBart, summarizeText, clean_text
import torch


trained_model = AraBart() 
trained_model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/summarize', methods = ['POST'])
def summary():
    text = request.form['text']
    cleaned_text = clean_text(text)
    result = list(map(lambda tex: summarizeText(tex, trained_model.model), cleaned_text))
    summary = '\n'.join(result)
    data = {'text': text, 'summary' : summary}
    return render_template('summary.html', data = data)



if __name__=="__main__":
    app.run(debug=True)




