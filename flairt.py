import flair
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
s = flair.data.Sentence("This is good idea")
flair_sentiment.predict(s)
total_sentiment = s.labels
print(total_sentiment)