import pandas as pd
df=pd.read_pickle('tweets_sentiment.pkl')

from tqdm.notebook import tqdm
import flair

ss= list(tqdm((flair.data.Sentence(sent) for sent in df['Tweet']), total=len(df)))

from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')
print("here")
tagger.predict(ss, verbose=True)
results = [ x.get_spans('ner') for x in ss]
df['NER'] = [x for x in results]
df.to_pickle('tweets_sentiment_ner.pkl')