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

tweets will be pulled like shows

![image](https://github.com/Refulic/EC601_tasks/blob/main/pics/tweet1.JPG)

![image](https://github.com/Refulic/EC601_tasks/blob/main/pics/tweet2.JPG)

