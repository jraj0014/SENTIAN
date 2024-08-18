#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#get_ipython().system('pip install sumy')


# In[2]:


# Installation of all the required packages
'''
import subprocess
import sys

# Function to install missing packages only
#def install(package):
#    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# List of required packages
required_packages = [
    'dash',
    'dash-core-components',
    'dash-html-components',
    'plotly',
    'requests',
    'nltk',
    'pandas',
    'wordcloud'
]

# Check and install packages
#for package in required_packages:
 #   try:
#    except ImportError:
 #       install(package)
#'''

import dash
import re
import plotly
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import matplotlib.font_manager as fm
#import plotly_dashboard
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
#nltk.download('punkt')
#nltk.download('vader_lexicon')
from datetime import datetime
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import base64

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

#Download VADER lexicon
#nltk.download('vader_lexicon')


# In[25]:


#API setup
NEWS_API_KEY = '0fe01a4650854ca7aae22271c17bcee9'
NEWS_BASE_URL = 'https://newsapi.org/v2/everything'
CRYPTO_API_URL = 'https://api.coingecko.com/api/v3/coins/{crypto}/market_chart/range'

#Function to fetch news articles
def get_news(query, start_date, end_date, language='en', page_size=100):
    params = {
        'q': query,
        'language': language,
        'pageSize': page_size,
        'apiKey': NEWS_API_KEY,
        'from': start_date,
        'to': end_date
    }
    response = requests.get(NEWS_BASE_URL, params=params)
    data = response.json()
    if response.status_code == 200:
        return data.get('articles', [])
    else:
        print("Error fetching news: {}".format(data.get('message', 'No error message')))
        return []

#Sumy Summariser.
def summarize_text(text, num_sentences=3, language="english"):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    lsasummarizer = LsaSummarizer()
    summary = lsasummarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in summary])

#Lex Rank Summariser.
def lexrank_summarize(text,summary_sentences=3, language='en'):
    # Create a parser for the input text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # Initialize the LexRank summarizer
    lexsummarizer = LexRankSummarizer()

    # Generate the summary
    summary = lexsummarizer(parser.document, num_sentences)

    # Join the sentences into a single string
    summary_text = " ".join(str(sentence) for sentence in summary)

    return summary_text

# Initialize the 'transformers' summarization pipeline.
#tfsummarizer = pipeline("summarization")

#Function to perform sentiment analysis on news articles
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    #print(query)
    sentiment_dict = sid.polarity_scores(text)
    sentiment = sentiment_dict['compound']
    if sentiment >= 0.05:
        return 'Positive'
    elif sentiment <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

#Function to fetch cryptocurrency data
def get_crypto_data(crypto, start_timestamp, end_timestamp):
    url = CRYPTO_API_URL.format(crypto=crypto)
    params = {
        'vs_currency': 'usd',
        'from': start_timestamp,
        'to': end_timestamp
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data.get('prices', [])
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to generate word cloud image
def generate_wordcloud(text, color, query):

    # Use a specific TrueType font file in your project directory
    #font_path = '/home/jraj1234/.fonts/LTYPE.TTF'
    # Use a default font that's likely to be available
    font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))

    # Create a set of words to exclude (convert to lowercase for case-insensitive comparison)
    stopwords = set(STOPWORDS)
    # Add the query and its variations to stopwords
    query_words = re.findall(r'\w+', query.lower())
    for word in query_words:
        stopwords.update([
            word,
            word + 's',  # plural
            word + 'ic',  # adjective form
            word[:-1] if word.endswith('s') else word,  # singular if plural was given
            word[:-2] if word.endswith('ic') else word  # root if adjective was given
        ])

    #Create the wordcloud.
    wordcloud = WordCloud(width=500, height=300, background_color='black', colormap=color,stopwords=stopwords,font_path=font_path).generate(text)
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = 'SENTIAN - News Sentiment Analysisand Crypto Correlation.'

#Import Datetime to set the start and end dates.
from datetime import datetime, timedelta

# Get current date and one week prior
current_date = datetime.now().date()
one_week_prior = current_date - timedelta(days=7)

# Define app layout
app.layout = html.Div([
     html.Div([
        html.H1('SENTIAN - Sentiment Analysis of News Articles',
                style={'color': 'black', 'padding': '10px'})
    ], style={'backgroundColor': 'peachpuff', 'marginBottom': '20px'}),
    html.Div([
        html.Div([
            html.H4("What's your favorite news topics?", style={'marginBottom': '8px'}),
            dcc.Input(
                id='news-query-input',
                type='text',
                value='',
                debounce=True,
                placeholder='Enter news topic. Eg: "Olympics"',
                style={'width': '100%'}
            ),
            dcc.Input(
                id='crypto-query-input',
                type='text',
                value='',
                debounce=True,
                placeholder='(Optional)Enter cryptocurrency for correlation.',
                style={'width': '100%'}  # Ensures full width
            ),
        ], style={'width': '40%', 'display': 'inline-block', 'padding': '0 10px'}),

        html.Div([
            html.H4("Select Date Range (Optional)", style={'marginBottom': '8px'}),
            dcc.DatePickerRange(
                id='date-picker',
                start_date=one_week_prior.strftime('%Y-%m-%d'),
                end_date=current_date.strftime('%Y-%m-%d'),
                display_format='YYYY-MM-DD',
                style={'width': '100%'}
            ),
            html.Button('Submit', id='submit-button', n_clicks=0, style={
                'backgroundColor': '#007bff',
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'fontSize': '16px',
                'cursor': 'pointer',
                'borderRadius': '5px',
                'marginTop': '10px'
            })
        ], style={'width': '40%', 'display': 'inline-block', 'padding': '0 10px'})
    ], style={'textAlign': 'center'}),

    html.Div(id='date-range-warning', style={'color': 'red', 'textAlign': 'center', 'margin': '20px'}),

    dcc.Graph(id='sentiment-bar-chart'),
    dcc.Graph(id='historical-trend-chart'),
    html.Div(id='article-list'),
  html.Div([
    html.H2('Summary of the News Articles', style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px 5px 0 0'}),
    html.Div(id='news-summary', style={
        'margin': '0',  # Changed from '20px'
        'padding': '20px',
        'border': '1px solid #ddd',
        'borderRadius': '0 0 5px 5px',
        'backgroundColor': '#ffffff'  # White background for the content
    })
    ], style={
    'backgroundColor': '#606060',  # Light gray background for the entire block
    'padding': '20px',
    'borderRadius': '5px',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'  # Optional: adds a subtle shadow
    }),
    html.Div([
    html.H2('Word Cloud for Positive Sentiments', style={'backgroundColor': '#66b266'}),
    html.Img(id='positive-wordcloud', style={'width': '80%', 'height': 'auto'}),
    html.H2('Word Cloud for Negative Sentiments', style={'backgroundColor': '#ff3232'}),
    html.Img(id='negative-wordcloud', style={'width': '80%', 'height': 'auto'})
    ]),

    #Add a loading spinner
    html.Div([
        dcc.Loading(
            id="loading",
            type="circle",
            children=[
                html.Div(id='spinner-content')
            ]
        )
    ])
])

#Defining a callback function
@app.callback(
    [Output('sentiment-bar-chart', 'figure'),
     Output('historical-trend-chart', 'figure'),
     Output('positive-wordcloud', 'src'),
     Output('negative-wordcloud', 'src'),
     Output('article-list', 'children'),
     Output('date-range-warning', 'children'),
     Output('news-summary', 'children')],  # Add this line
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('news-query-input', 'value'),
     dash.dependencies.State('crypto-query-input', 'value'),
     dash.dependencies.State('date-picker', 'start_date'),
     dash.dependencies.State('date-picker', 'end_date')]
)
def update_output(n_clicks, news_query, crypto_query, start_date, end_date):
    if n_clicks > 0 and news_query:
        try:
            #Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

            #Check date range
            date_range = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
            if date_range > 20:
                return {}, {}, '', '', html.Div(
                    'Date range should be within 20 days to current date.'), 'Date range should be within 20 days to current date.'
        except ValueError as e:
            return {}, {}, '', '', html.Div('Error parsing dates: {}'.format(str(e))),''


        #Fetch and analyze news data using the get_news function
        articles = get_news(news_query, start_date=start_date, end_date=end_date)
        #if not articles:
        #    return {}, {}, '', '', html.Div('No articles found.'), ''

        sent_cat = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        daily_sentiment = {}
        #query =''
        positive_text = ''
        negative_text = ''
        article_elements = []

          #Get summary.
        all_text = " ".join([article['description'] for article in articles if article['description']])
        summary = summarize_text(all_text)

        for i, article in enumerate(articles):
            title = article['title']
            description = article['description']
            url = article['url']
            published_at = article['publishedAt']

            if description:
                sentiment = analyze_sentiment(description)
                if sentiment == 'Positive':
                    positive_text += ' ' + description
                elif sentiment == 'Negative':
                    negative_text += ' ' + description
            else:
                sentiment = 'Neutral'

            if sentiment in sent_cat:
                sent_cat[sentiment] += 1

            # Track sentiment over time
            date = published_at.split('T')[0]
            if date not in daily_sentiment:
                daily_sentiment[date] = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
            daily_sentiment[date][sentiment] += 1

            # Article list for the UI
            # article_elements.append(html.Div([
            #     html.H4(f"{i + 1}. {title}"),
            #     html.P(f"Description: {description}"),
            #     html.P(f"Sentiment: {sentiment}"),
            #     html.A("Read more", href=url, target="_blank"),
            #     html.Hr()
            # ]))

        #Generate word clouds
        positive_wordcloud = generate_wordcloud(positive_text, 'Greens', news_query)
        negative_wordcloud = generate_wordcloud(negative_text, 'Reds', news_query)

        #Create sentiment bar chart
        sentiment_fig = go.Figure(data=[go.Bar(
            x=list(sent_cat.keys()),
            y=list(sent_cat.values()),
            marker_color=['green', 'gray', 'red']
        )])
        sentiment_fig.update_layout(
            title={
        'text': '<b>Sentiment Analysis of News Articles</b>',
        'font': {'size': 24}
        },
            xaxis_title='Sentiment',
            yaxis_title='Number of Articles'
        )

        #Create historical trend chart
        historical_trend_fig = go.Figure()

        #Add news sentiment trend traces
        dates = sorted(daily_sentiment.keys())
        pos_values = [daily_sentiment[date]['Positive'] for date in dates]
        neu_values = [daily_sentiment[date]['Neutral'] for date in dates]
        neg_values = [daily_sentiment[date]['Negative'] for date in dates]

        historical_trend_fig.add_trace(
            go.Scatter(x=dates, y=pos_values, mode='lines+markers', name='Positive Sentiment', line=dict(color='green'))
        )
        historical_trend_fig.add_trace(
            go.Scatter(x=dates, y=neu_values, mode='lines+markers', name='Neutral Sentiment', line=dict(color='gray'))
        )
        historical_trend_fig.add_trace(
            go.Scatter(x=dates, y=neg_values, mode='lines+markers', name='Negative Sentiment', line=dict(color='red'))
        )

        #Add cryptocurrency price trace if crypto_query is provided
        if crypto_query:
            crypto_df = get_crypto_data(crypto_query, start_timestamp, end_timestamp)
            historical_trend_fig.add_trace(go.Scatter(
                x=crypto_df['date'],
                y=crypto_df['price'],
                mode='lines',
                name='{} Price'.format(crypto_query.capitalize()),
                yaxis='y2',
                line=dict(color='blue')
            ))

        #Update layout for historical trend plot
        historical_trend_fig.update_layout(
            title={
            'text': '<b>Historical Trend of News Sentiment and {}</b>'.format(crypto_query.capitalize() if crypto_query else "without Crypto Data"),
            'font': {'size': 24}
            },
            xaxis_title='Date',
            yaxis_title='Number of Articles',
            yaxis2=dict(
                title='Price (USD)',
                overlaying='y',
                side='right'
            )
        )

        return (sentiment_fig, historical_trend_fig,
        "data:image/png;base64,{}".format(positive_wordcloud),
        "data:image/png;base64,{}".format(negative_wordcloud),
        article_elements, '', summary)

    #Default values in the search box
    return {}, {}, '', '', html.Div('Enter a search query to see the results.'),'',''

#Staring the app server
if __name__ == '__main__':
    app.run_server(debug=True)



# In[ ]:





# In[ ]:





# In[ ]:




