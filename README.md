# EC601_tasks
Sentiment Analysis with Tweepy
========
Set up environment
--

```
import tweepy
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')

import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
```

Twitter Authentication
--
Register twitter API and get keys to generate Bearer Token.


![image](https://github.com/Refulic/EC601_tasks/blob/main/pics/keys_and_tokens.JPG)

Get recent public tweets
--

Pull tweets from twitter

Get tweets that contain the hashtag

In this file, use the tag "facebook"

tweets will be pulled like the following shows:

![image](https://github.com/Refulic/EC601_tasks/blob/main/pics/tweet1.JPG)

![image](https://github.com/Refulic/EC601_tasks/blob/main/pics/tweet2.JPG)

Data processing to generate Sentiment Labels
--
Calculating Negative, Positive, Neutral and Compound values

![image](https://github.com/Refulic/EC601_tasks/blob/main/pics/tweet2.JPG)

Sentiment Visualisation
--
create data for Donut Chart

This char will directly shows the different sentiments.
![image]([https://github.com/Refulic/EC601_tasks/blob/main/pics/tweet2.JPG](https://github.com/Refulic/EC601_tasks/blob/main/pics/graphy.JPG))




