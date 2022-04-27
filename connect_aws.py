import boto3
import json

#initialize comprehend module
comprehend = boto3.client(service_name='comprehend', region_name='us-east-1')




documents = ['My lovely Pat has one of the GREAT voices of her generation. I have listened to this CD for YEARS and I still LOVE IT. When I\'m in a good mood it makes me feel better. A bad mood just evaporates like sugar in the rain. This CD just oozes LIFE. Vocals are jusat STUUNNING and lyrics just kill. One of life\'s hidden gems. This is a desert isle CD in my book. Why she never made it big is just beyond me. Everytime I play this, no matter black, white, young, old, male, female EVERYBODY says one thing "Who was that singing ?"', "Reviewed quite a bit of the combo players and was hesitant due to unfavorable reviews and size of machines. I am weaning off my VHS collection, but don't want to replace them with DVD's. This unit is well built, easy to setup and resolution and special effects (no progressive scan for HDTV owners) suitable for many people looking for a versatile product.Cons- No universal remote."]
res = []

#actual sentiment analysis loop
for id, comments in enumerate(documents):
    # here is the main part - comprehend.detect_sentiment is called
    sentimentData = comprehend.detect_sentiment(Text=comments, LanguageCode='en')
    # # preparation of the data for the insert query 
    qdata = {
    'id': id,
    'Sentiment': "ERROR",
    'MixedScore': 0,
    'NegativeScore': 0,
    'NeutralScore': 0,
    'PositiveScore': 0,
    }
    if 'Sentiment' in sentimentData:
        qdata['Sentiment'] = sentimentData['Sentiment']
    if 'SentimentScore' in sentimentData:
        if 'Mixed' in sentimentData['SentimentScore']:
            qdata['MixedScore'] = sentimentData['SentimentScore']['Mixed']
        if 'Negative' in sentimentData['SentimentScore']:
            qdata['NegativeScore'] = sentimentData['SentimentScore']['Negative']
        if 'Neutral' in sentimentData['SentimentScore']:
            qdata['NeutralScore'] = sentimentData['SentimentScore']['Neutral']
        if 'Positive' in sentimentData['SentimentScore']:
            qdata['PositiveScore'] = sentimentData['SentimentScore']['Positive']
    res.append(qdata)

for (id, comments) in enumerate(documents):
    # print(comments)
    print(res[id])

