import snscrape.modules.twitter as sntwitter
import pandas as pd

# query = "(from:elonmusk) until:2020-01-01 since:2010-01-01"
query = '"ukraine"  ("war" OR or OR "operation" OR or OR "military" OR or OR "putin") min_faves:10 min_retweets:10 until:2022-04-27 since:2022-01-14'
tweets = []
limit = 10000000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # # print(vars(tweet))
    # # break
    # if(len(tweets)%10000==0):
    #     print(len(tweets))
    # if len(tweets) == limit:
    #     break
    # else:
        # print(vars(tweet))
    tweets.append([tweet.date.date(), tweet.user.location, tweet.user.username, tweet.content])
        
df = pd.DataFrame(tweets, columns=['Date', 'Location', 'User', 'Tweet'])
df.to_csv("tweets.csv")
print(len(df))
