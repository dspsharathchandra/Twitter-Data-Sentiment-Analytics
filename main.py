# import time
# start_time = time.time()
import pandas as pd
pd.options.mode.chained_assignment = None
import string
import re
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(60)
main_dict={}
vec=[]
sw=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't","a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", ""," ", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "were", "werent", "what", "whats", "when", "whens", "where", "\n","\r","wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
def generate_ngrams(text,n):
    tokens=re.split("\\s+",text)
    ngrams=[]
    for i in range(len(tokens)-n+1):
       temp=[tokens[j] for j in range(i,i+n)]
       ngrams.append(" ".join(temp))

    return ngrams

def remove_stopwords(text):
    text=text.lower()
    text1 = re.sub("[^\w]", " ",  text).split()
    for word in text1:
        if word in sw:
            text1.remove(word)
    return " ".join(text1)

def remove_punctuation(text):
  if(type(text)==float):
    return text
  ans=""  
  for i in text:     
    if i not in string.punctuation:
      ans+=i    
  return ans
def remove_nl(text):
    return text.replace('\r', '').replace('\n', '')

def generate_dict(ngrams):
    for i in ngrams:
        if(i in main_dict):
            main_dict[i]=main_dict[i]+1
        else:
            main_dict[i]=1
def vectorize(ngram):
    n_dict={}
    for elem in ngram:
        if(elem not in n_dict):
            n_dict[elem]=1
        else:
            n_dict[elem]=n_dict[elem]+1
    n_vect=[]
    for i in vec:
        if i in n_dict:
            n_vect.append(n_dict[i])
        else:
            n_vect.append(0)
    return n_vect

def accuracy(orig, pred):
    return np.mean(orig==pred)


class Logistic():
    def __init__(self):
        self.weights = None
        self.bias = None


    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))



    def train(self, labeled_data, learning_rate, max_epochs, lam, reg_method='L2'):
        # print("inside train")
        X_train, Y_train=labeled_data['ngram_vector'], labeled_data['emotions']
        # X_test, Y_test=test_data['ngram_vector'], test_data['Label']
        X_train = X_train.values
        Y_train = Y_train.values
        rv=X_train[0]
        for i in X_train[1:]:
            rv=np.vstack((rv,i))
        X_train=rv
        X_train = X_train.T
        Y_train = Y_train.reshape(1, X_train.shape[1])

    
        m = X_train.shape[1]
        n = X_train.shape[0]
        X=np.array(X_train, dtype=np.float64)
        Y=np.array(Y_train, dtype=np.float64)
        
        self.weights = np.zeros((n,1))
        self.bias = 0
        
        
        for i in range(max_epochs):
            Z = np.array(np.dot(self.weights.T, X[:,:]), dtype=np.float64) + self.bias
            A = np.array(self.sigmoid(Z), dtype=np.float64)
            if(reg_method=='L2'):
                dW = np.dot(A-Y[:,:], X[:,:].T)+lam*np.sum(np.square(self.weights/m))
                dB = np.sum(A - Y[:,:])+lam*np.sum(np.square(self.weights/m))
            elif(reg_method=='L1'):
                dW = np.dot(A-Y[:,:], X[:,:].T)+lam*np.sum(self.weights/m)
                dB = np.sum(A - Y[:,:])+lam*np.sum(self.weights/m)
            
            self.weights = self.weights - learning_rate*dW.T
            self.bias = self.bias - learning_rate*dB
        return self.weights, self.bias
        
    
    
    def predict(self, data,g_weights,g_bias):
        predicted_labels = []
        X=data['ngram_vector']
        X = X.values
        rv=X[0]
        for i in X[1:]:
            rv=np.vstack((rv,i))
        X=rv
        X = X.T
        Z = np.dot(g_weights.T, X) + g_bias
        predicted_labels = self.sigmoid(Z)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels.T

class Neural_Network():
    def __init__(self, ):
        pass

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self,x):
        return self.sigmoid(x) *(1-self.sigmoid (x))

    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)
    def train(self,X_train,y_train,X_test,Y_test, test_data_unseen,learning_rate, iterations):
        np.random.seed(0)
        data = X_train
        labels = y_train
        number_of_examples = data.shape[0]
        features = data.shape[1]
        h_layer = 64
        o_layer = 6
        weights_hidden = np.random.rand(features,h_layer)
        bias_hidden = np.random.randn(h_layer)
        weights_output = np.random.rand(h_layer,o_layer)
        bias_output = np.random.randn(o_layer)
        for _ in range(iterations):
            ff1 = np.dot(data, weights_hidden) + bias_hidden
            o1 = self.sigmoid(ff1)
            ff2 = np.dot(o1, weights_output) + bias_output
            o2 = self.softmax(ff2)
            do1 = o2 - labels
            do2 = o1
            do3 = np.dot(do2.T, do1)
            do4 = do1
            dh1 = weights_output
            dh2 = np.dot(do1 , dh1.T)
            dh3 = self.sigmoid_der(ff1)
            dh4 = data
            d_wh = np.dot(dh4.T, dh3 * dh2)
            d_bh = dh2 * dh3

            weights_hidden -= learning_rate * d_wh 
            bias_hidden -= learning_rate * d_bh.sum(axis=0)

            weights_output -= learning_rate * do3
            bias_output -= learning_rate * do4.sum(axis=0)


        zh = np.dot(X_train, weights_hidden) + bias_hidden
        ah = self.sigmoid(zh)
        zo = np.dot(ah, weights_output) + bias_output
        ao = self.softmax(zo)
        o=np.argmax(ao,axis=1)
        train_accuracy=np.mean(o==np.argmax(labels,axis=1))
        # print('Train accuracy:',train_accuracy)
        zh = np.dot(X_test, weights_hidden) + bias_hidden
        ah = self.sigmoid(zh)
        zo = np.dot(ah, weights_output) + bias_output
        ao = self.softmax(zo)
        o=np.argmax(ao,axis=1)
        Y_test_s=np.resize(Y_test,(240,))
        test_accuracy = np.mean(o==Y_test_s)
        # print('Test accuracy:', test_accuracy)
        X_test=test_data_unseen['ngram_vector']
        # X_test, Y_test=test_data['ngram_vector'], test_data['Label']
        X_test_unseen = X_test.values
        rv=X_test_unseen[0]
        for i in X_test_unseen[1:]:
            rv=np.vstack((rv,i))
        X_test_unseen=rv
        # X_train = X_train.T
        zh = np.dot(X_test_unseen, weights_hidden) + bias_hidden
        ah = self.sigmoid(zh)
        zo = np.dot(ah, weights_output) + bias_output
        ao = self.softmax(zo)
        o=np.argmax(ao,axis=1)
        # print('Test data unseen labels:',o)
        l=['joy','sadness','anger','fear','love','surprise']
        oo=[]
        for ii,elem in enumerate(o):
            oo.append(l[elem])
            # o[ii]=l[elem]
        test_data_unseen_final = pd.read_csv('test.csv', index_col=0)
        test_data_unseen_final['emotions'] = oo
        test_data_unseen_final.to_csv('test_nn.csv')

        



def LR(train_data,test_data,test_data_unseen,learning_rate, max_epochs, lam,reg_method):
    logistic = Logistic()
    l=['joy','sadness','anger','fear','love','surprise']
    weights=[]
    bias=[]
    test_data_x=test_data.copy()
    train_data_x=train_data.copy()
    for i,e in enumerate(l):
        train_data_x.loc[train_data_x.emotions == e, 'emotions']=i
        test_data_x.loc[test_data_x.emotions == e, 'emotions']=i

    for n,emo in enumerate(l):
        # print("For emotion:", emo)
        df_emo=train_data.copy()
        df_emo.loc[df_emo.emotions == emo, 'emotions']=1
        df_emo.loc[df_emo.emotions != 1, 'emotions']=0
    #     print(df_emo['emotions'].value_counts())
        logistic = Logistic()
        w,b=logistic.train(df_emo,learning_rate, max_epochs, lam,reg_method)
        # w,b=logistic.train(labeled_data=df_emo,learning_rate=0.01, max_epochs=100, lam=0.1,reg_method='L2')
        weights.append(w)
        bias.append(b)
        # out=logistic.predict(df_emo,weights[n],bias[n])
        # accuracy(df_emo['emotions'].tolist(),out)
    outp_train=np.array([])
    outp_test=np.array([])
    outp_test_unseen=np.array([])
    for i,emo in enumerate(l):
        out_train=logistic.predict(train_data,weights[i],bias[i])
        out_test=logistic.predict(test_data,weights[i],bias[i])
        out_test_unseen=logistic.predict(test_data_unseen,weights[i],bias[i])
        if(i==0):
            outp_train=out_train
            outp_test=out_test
            outp_test_unseen=out_test_unseen
        else:
            outp_train=np.hstack((outp_train,out_train))
            outp_test=np.hstack((outp_test,out_test))
            outp_test_unseen=np.hstack((outp_test_unseen,out_test_unseen))
        # print(outp_train.shape)
        # print(outp_test.shape)
        # print(outp_test_unseen.shape)
    outpx_train = np.true_divide(outp_train, outp_train.sum(axis=1, keepdims=True))
    res=outpx_train.argmax(axis=1)
    train_accuracy = accuracy(train_data_x['emotions'].tolist(),res)
    # print("Train Accuracy:",train_accuracy)
    
    outpx_test = np.true_divide(outp_test, outp_test.sum(axis=1, keepdims=True))
    res=outpx_test.argmax(axis=1)
    test_accuracy = accuracy(test_data_x['emotions'].tolist(),res)
    # print("Test Accuracy:", test_accuracy)

    outpx_test_unseen = np.true_divide(outp_test_unseen, outp_test_unseen.sum(axis=1, keepdims=True))
    o=outpx_test_unseen.argmax(axis=1)
    # test_accuracy = accuracy(test_data_x['emotions'].tolist(),res)
    # print("Test unseen labels:", o)
    l=['joy','sadness','anger','fear','love','surprise']
    oo=[]
    for ii,elem in enumerate(o):
        oo.append(l[elem])
    test_data_unseen_final = pd.read_csv('test.csv', index_col=0)
    test_data_unseen_final['emotions'] = oo
    test_data_unseen_final.to_csv('test_lr.csv')

    

    # your logistic regression 

def NN(X_train,y_train,X_test,Y_test, test_data_unseen,learning_rate,iterations):
    nn = Neural_Network()
    nn.train(X_train,y_train,X_test,Y_test, test_data_unseen,learning_rate,iterations)

    # your Multi-layer Neural Network

if __name__ == '__main__':
    all_data = pd.read_csv('train.csv', index_col=0)
    all_data['text']= all_data['text'].apply(lambda x:remove_nl(x))
    all_data['text_']= all_data['text'].apply(lambda x:remove_punctuation(x))
    all_data['text_sw']= all_data['text_'].apply(lambda x:remove_stopwords(x))
    all_data['text_sw']= all_data['text_sw'].apply(lambda x:remove_stopwords(x))
    all_data['text_sw']= all_data['text_sw'].apply(lambda x:remove_stopwords(x))
    all_data['text_ng']= all_data['text_sw'].apply(lambda x:generate_ngrams(x.strip(),1)) # 2 gram is selected
    all_data['text_ng'].apply(lambda x:generate_dict(x))
    vec = list(main_dict.keys()) #This is the overall vocabulary present in the train set
    # selected_set=dict(sorted(main_dict.items(), key=lambda x: x[1], reverse=True)[0:1000])
    # vec=list(selected_set.keys()) #This is the this is the vocabulary we selected for test
    all_data['ngram_vector']= all_data['text_ng'].apply(lambda x:vectorize(x))
    all_data=all_data.drop(columns=['text_','text_sw','text_ng','text'])

    test_data_unseen = pd.read_csv('test.csv', index_col=0)
    test_data_unseen['text']= test_data_unseen['text'].apply(lambda x:remove_nl(x))
    test_data_unseen['text_']= test_data_unseen['text'].apply(lambda x:remove_punctuation(x))
    test_data_unseen['text_sw']= test_data_unseen['text_'].apply(lambda x:remove_stopwords(x))
    test_data_unseen['text_sw']= test_data_unseen['text_sw'].apply(lambda x:remove_stopwords(x))
    test_data_unseen['text_sw']= test_data_unseen['text_sw'].apply(lambda x:remove_stopwords(x))
    test_data_unseen['text_ng']= test_data_unseen['text_sw'].apply(lambda x:generate_ngrams(x.strip(),1)) # # 1 gram is selected
    test_data_unseen['ngram_vector']= test_data_unseen['text_ng'].apply(lambda x:vectorize(x))
    test_data_unseen=test_data_unseen.drop(columns=['text_','text_sw','text_ng','text'])

    test_data = all_data.sample(frac=0.2)
    train_data = all_data.drop(test_data.index)

    


    
    
    
    print ("..................Beginning of Logistic Regression................")

    LR(train_data, test_data, test_data_unseen, 0.01, 700, 0.1,'L2')
    print ("..................End of Logistic Regression................")

    print("------------------------------------------------")
    l=['joy','sadness','anger','fear','love','surprise']
    for i,li in enumerate(l):
        train_data.loc[train_data.emotions == li, 'emotions']=i
        test_data.loc[test_data.emotions == li, 'emotions']=i
    X_train, Y_train=train_data['ngram_vector'], train_data['emotions']
    # X_test, Y_test=test_data['ngram_vector'], test_data['Label']
    X_train = X_train.values
    Y_train = Y_train.values
    rv=X_train[0]
    for i in X_train[1:]:
        rv=np.vstack((rv,i))
    X_train=rv
    # X_train = X_train.T
    Y_train = Y_train.reshape(X_train.shape[0],1)
    X_test, Y_test=test_data['ngram_vector'], test_data['emotions']
    # X_test, Y_test=test_data['ngram_vector'], test_data['Label']
    X_test = X_test.values
    Y_test = Y_test.values
    rv=X_test[0]
    for i in X_test[1:]:
        rv=np.vstack((rv,i))
    X_test=rv
    # X_train = X_train.T
    Y_test = Y_test.reshape(X_test.shape[0],1)
    y_train = np.zeros((Y_train.shape[0], 6))
    for i in range(Y_train.shape[0]):
        y_train[i, Y_train[i][0]] = 1
    y_test = np.zeros((Y_test.shape[0], 6))

    for i in range(Y_test.shape[0]):
        y_test[i, Y_test[i][0]] = 1

    print ("..................Beginning of Neural Network................")
    NN(X_train,y_train,X_test,Y_test, test_data_unseen,10e-4,5000)
    print ("..................End of Neural Network................")
    # print("--- %s seconds ---" % (time.time() - start_time))



## This is the code for hyper parameters
# train_accuracies = []
# test_accuracies = []
# iters = [1,10,30, 50, 100,200,300,400,500,600,700,800,900,1000]
# for iter in iters:
#     train_accuracy, test_accuracy = LR(train_data, test_data,0.01, iter, 0.1,'L2')
#     train_accuracies.append(train_accuracy)
#     test_accuracies.append(test_accuracy)
# plt.plot(iters,train_accuracies,label="Training Set")
# plt.plot(iters,test_accuracies,label="Validation Set")
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.title("LR Cross validation with Iterations")
# plt.legend()
# plt.show()

# lrs = [0.0000001,0.0000005,0.000001,0.000005,0.00001,0.0001,0.0005, 0.0010, 0.0050, 0.010,0.050, 0.1]#200,300,400,500,600,700,800,900,1000]
# for lr in lrs:
#     train_accuracy, test_accuracy = LR(train_data, test_data, lr, 700, 0.1,'L2')
#     train_accuracies.append(train_accuracy)
#     test_accuracies.append(test_accuracy)
# plt.plot(lrs,train_accuracies,label="Training Set")
# plt.plot(lrs,test_accuracies,label="Validation Set")
# plt.xlabel('Learning rate')
# plt.ylabel('Accuracy')
# plt.title("LR Cross validation with Learning rate(Iters=700)")
# plt.legend()
# plt.show()