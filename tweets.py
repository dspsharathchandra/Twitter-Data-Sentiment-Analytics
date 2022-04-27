import snscrape.modules.twitter as sntwitter
import pandas as pd

# query = "(from:elonmusk) until:2020-01-01 since:2010-01-01"
query = '"ukraine" and "russia"  ("war" OR or OR "operation" OR or OR "military" OR or OR "putin") min_faves:60 min_retweets:40 until:2022-03-30 since:2022-02-14'
tweets = []
limit = 1000000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        # print(vars(tweet))
        tweets.append([tweet.date, tweet.user.location, tweet.user.username, tweet.content])
        
df = pd.DataFrame(tweets, columns=['Date', 'Location', 'User', 'Tweet'])
df.to_csv("tweets.csv")
print(df)
