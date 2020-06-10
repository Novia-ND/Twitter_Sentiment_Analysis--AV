#!/usr/bin/env python
# coding: utf-8

# # Importing The DataSets

# In[48]:


get_ipython().run_line_magic('pip', 'install wordcloud')


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


dataset=pd.read_csv('train_E6oV3lV (1).csv')
dataset2=pd.read_csv('test_tweets_anuFYb8.csv')
X_test = dataset2.values


# In[3]:


dataset


# In[4]:


dataset = dataset.drop(['id'],axis = 1)


# In[5]:


dataset


# In[6]:


y = dataset.iloc[:,:-1].values #independent variable


# # Checking For Missing Values

# In[7]:


dataset.info()


# # Analysing the Data

# In[8]:


print(dataset.head())


# In[9]:


print(y)


# In[10]:


ham = dataset[dataset['label'] == 0] 
spam = dataset[dataset['label'] == 1 ]
length = [len(ham),len(spam)]
print(length)


# # Visualizing The Data

# In[11]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
label = ['ham','spam']
ax.bar(label,length)
plt.show()


# In[12]:


count_Class=pd.value_counts(dataset["label"], sort= True)
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('label')
plt.show()


# In[13]:


from collections import Counter
count1 = Counter(" ".join(dataset[dataset['label']== 0]["tweet"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
count2 = Counter(" ".join(dataset[dataset['label']== 1]["tweet"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})


# In[14]:


df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


# In[15]:


df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


# # For Train Set 

# #  Cleaning the texts and steming it 

# In[16]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 31962 ):
  review = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# In[17]:


print(corpus)


# # Creating the Bag of Words model for train set
# 
# 

# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=all_stopwords, max_features = 1000)
print(vectorizer)
X_train = vectorizer.fit_transform(corpus).toarray()
print(X_train)


# # Creating a DocumentTermMatrix

# In[19]:


df2 = pd.DataFrame(X_train.transpose(),
                   index=vectorizer.get_feature_names())
print(df2)


# # Making a WordCloud from Corpus Data

# In[49]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = all_stopwords

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=all_stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))

    plt.imshow(wordcloud)

show_wordcloud(corpus)


# # Applying KMeans Clustering Algorithm

# In[51]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  k_means=KMeans(n_clusters = i, init="k-means++", random_state=42)
  k_means.fit(X_train)
  wcss.append(k_means.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
from sklearn.cluster import MiniBatchKMeans


# In[21]:


from sklearn.cluster import MiniBatchKMeans
cls = MiniBatchKMeans(n_clusters=5, random_state=0)
cls.fit(X_train)
cls.predict(X_train)
cls.labels_


# # For Test Set

# In[22]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus2 = []
for i in range(0, 17197):
  review2 = re.sub('[^a-zA-Z]', ' ', dataset2['tweet'][i])
  review2 = review2.lower()
  review2 = review2.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review2 = [ps.stem(word) for word in review2 if not word in set(all_stopwords)]
  review2 = ' '.join(review2)
  corpus2.append(review2)


# In[23]:


print(corpus2)


# # Creating the Bag of Words model for test set
# 

# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer2 = TfidfVectorizer(stop_words=all_stopwords, max_features = 1000)
print(vectorizer2)
X_test = vectorizer2.fit_transform(corpus2).toarray()
print(X_test)


# # Creating a DocumentTermMatrix

# In[25]:


df2 = pd.DataFrame(X_test.transpose(),
                   index=vectorizer2.get_feature_names())
print(df2)


# # Making a WordCloud from Corpus Data

# In[50]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = all_stopwords

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=all_stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))

    plt.imshow(wordcloud)

show_wordcloud(corpus2)


# # Linear dimensionality reduction

# In[27]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)
reduced_features = pca.fit_transform(X_train)
print(reduced_features)
reduced_cluster_centers = pca.transform(cls.cluster_centers_)


# # Visualizing The Clusters

# In[28]:


plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(X_train))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')


# # Splitting The Dataset into training and testing

# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_train,y,test_size=0.2, random_state=0)


# In[30]:


x_train


# ## Testing different Models

# # Logistic Regression

# In[31]:


from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)


# In[32]:


y_pred=log_reg.predict(x_test)
print(y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Blues'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['ham', 'spam']); ax.yaxis.set_ticklabels(['ham', 'spam']);


# # Random Forest Classifier

# In[34]:


from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
Classifier.fit(x_train,y_train)


# In[35]:


y_pred = Classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[36]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='rainbow'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['ham', 'spam']); ax.yaxis.set_ticklabels(['ham', 'spam']);


# #Naive Bayes Classifier

# In[38]:


from sklearn.naive_bayes import GaussianNB
clas=GaussianNB()
clas.fit(x_train,y_train)


# In[39]:


y_pred=clas.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[40]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_pred,y_test)
print(cm)
accuracy_score(y_pred, y_test)


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='flag'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['ham', 'spam']); ax.yaxis.set_ticklabels(['ham', 'spam']);


# ## Selected Logistic Regression Classifier

# # Prediction on Manual Inputs
# 
# 
# 

# In[42]:


p = [input("enter a string::")]
p = vectorizer.transform(p).toarray()
print(p)


# In[43]:


print(p.shape[1], X_train.shape[1])


# In[45]:


if log_reg.predict(p) == 0:
  print("Not Spam")
else :
  print(Spam)


# # Predicting the Test File

# In[46]:


Y_Pred_Test = log_reg.predict(X_test)
Y_Pred_Test


# ## Dumping the model as using joblib

# In[47]:


import joblib
filename1 = 'finalized_model4.sav'
filename2 = 'finalized_model5.sav'
joblib.dump(log_reg, filename1)
joblib.dump(vectorizer, filename2)

