from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

credential = AzureKeyCredential("62c5e36b53944bf09eb953445d6d2a7e")
text_analytics_client = TextAnalyticsClient(endpoint="https://cloudprojnlp.cognitiveservices.azure.com/", credential=credential)
documents = ['My lovely Pat has one of the GREAT voices of her generation. I have listened to this CD for YEARS and I still LOVE IT. When I\'m in a good mood it makes me feel better. A bad mood just evaporates like sugar in the rain. This CD just oozes LIFE. Vocals are jusat STUUNNING and lyrics just kill. One of life\'s hidden gems. This is a desert isle CD in my book. Why she never made it big is just beyond me. Everytime I play this, no matter black, white, young, old, male, female EVERYBODY says one thing "Who was that singing ?"', "The movie made it into my top ten favorites. What a great movie!"]

result = text_analytics_client.analyze_sentiment(documents, show_opinion_mining=True)
docs = [doc for doc in result if not doc.is_error]
print(docs)
print("Let's visualize the sentiment of each of these documents")
for idx, doc in enumerate(docs):
    print("Overall sentiment: {}".format(doc))
    print("Overall sentiment: {}".format(doc.confidence_scores.negative))
    print("Overall sentiment: {}".format(doc.sentiment))
