from flask import Flask, request, render_template
import requests
from datetime import datetime
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dotenv import load_dotenv
import os

# Load the NER model and tokenizer
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)


load_dotenv()  
api_key = os.getenv('API_KEY')

app = Flask(__name__)

# Load the FinBERT model and tokenizer
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# Initialize the sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

def construct_api_url(tickers, date, api_key):
    formatted_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&time_from={formatted_date}T0000&time_to={formatted_date}T2359&apikey={api_key}"
    return url


#Instrcutions provided from www.alphavantage.co API instructions
def fetch_financial_news(url):
    response = requests.get(url)
    if response.status_code != 200:
        return []

    news_data = response.json()
    return [article['title'] for article in news_data.get('feed', [])]

@app.route('/', methods=['GET', 'POST'])
def index():
    news_titles = []
    sentiments = []
    api_url = ''
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        date = request.form.get('date')
        if ticker and date:
            api_url = construct_api_url(ticker, date, api_key)
            news_titles = fetch_financial_news(api_url)
            sentiments = [nlp(title)[0] for title in news_titles]
        context = {
            'selected_ticker': ticker,
            'news_data': zip(news_titles, sentiments),
            'selected_date': date,
            'api_url': api_url  
        }
    else:
        context = {'news_data': [], 'api_url': api_url}

    entity_sentiments = analyze_titles_for_ner_and_sentiment(news_titles)
    context['entity_sentiments'] = entity_sentiments
    
    return render_template('index.html', **context)

def update_entity_sentiment(entity_sentiment_count, entity, title):
    sentiment_result = nlp(title)[0]['label']
    
    if entity not in entity_sentiment_count:
        entity_sentiment_count[entity] = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Total': 0}
    entity_sentiment_count[entity][sentiment_result] += 1
    entity_sentiment_count[entity]['Total'] += 1


# Use the NER pipeline to indentify people and orgs
# parsing code adapted from https://huggingface.co/dslim/bert-base-NER
def analyze_titles_for_ner_and_sentiment(titles):
    entity_sentiment_count = {}
    for title in titles:
        ner_results = ner_pipeline(title)

        full_entity = ""
        entity_type = ""
        for ner in ner_results:
            entity_type = ner['entity'][2:] 
            if ner['entity'].startswith('B-'):
                if full_entity and (entity_type == 'ORG' or entity_type == 'PER'):
                    update_entity_sentiment(entity_sentiment_count, full_entity, title)
                full_entity = ner['word']
            elif ner['entity'].startswith('I-') and full_entity:
                full_entity += " " + ner['word']

        if full_entity and (entity_type == 'ORG' or entity_type == 'PER'):
            update_entity_sentiment(entity_sentiment_count, full_entity, title)

    # Only track entites above 3 to avoid clutter and maintainly meanfulness
    filtered_entities = {k: v for k, v in entity_sentiment_count.items() if v['Total'] >= 3}
    return filtered_entities

if __name__ == '__main__':
    app.run(debug=True)
