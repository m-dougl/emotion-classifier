import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import unidecode
import nltk

from nltk.tokenize import RegexpTokenizer
from unicodedata import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

nltk.download('stopwords')

warnings.filterwarnings('ignore')
dataset = pd.read_excel('base_tres_emocoes.xlsx')
df=pd.DataFrame(dataset)

stopwords_nltk=nltk.corpus.stopwords.words('portuguese')
regexp=RegexpTokenizer('\w+')

df_features=df['Comentarios'].str.lower()
def remove_accents(text):
    return normalize('NFKD', text).encode('ASCII','ignore').decode('ASCII')
df_features=df_features.apply(remove_accents)
df_features=df_features.apply(regexp.tokenize)
df_features=df_features.apply(lambda x: [i for i in x if i not in stopwords_nltk])
df_features=df_features.apply(lambda x: ' '.join(i for i in x if len(i)>2))

df_labels=df['Emoção'].str.lower()
df_labels=LabelEncoder().fit_transform(df_labels)


X_train,X_test,y_train,y_test=train_test_split(df_features,df_labels,test_size=0.2,stratify=df_labels,random_state=42)

tfidfvectorizer=TfidfVectorizer()

X_train=tfidfvectorizer.fit_transform(X_train.ravel())
X_test=tfidfvectorizer.transform(X_test.ravel())

# Create a ML models
svm_parameters={'C': [0.01,0.1,1,10,100,1000],
                'kernel':['linaer','poly','rbf','sigmoid'],
                'gamma':[0.001,0.01,0.1,1,10,100],
                'probability' : [True]       
                }

nb_parameters={'alpha':[0.01,0.1,1,10,100,1000],
               'fit_prior': [True,False]
              }

knn_parameters={'n_neighbors':[3,4,5,6,7,8,9,10,11,12,13,14,15],
                'algorithm':['auto','ball_tree','kd_tree','brute'],
                'p':[2,3,4,5]
               }

clf_svm=GridSearchCV(estimator=SVC(), param_grid=svm_parameters).fit(X_train,y_train)
clf_nb=GridSearchCV(estimator=MultinomialNB(), param_grid=nb_parameters).fit(X_train,y_train)
clf_knn=GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_parameters).fit(X_train,y_train)

svm_predict= clf_svm.predict(X_test)
nb_predict= clf_nb.predict(X_test)
knn_predict= clf_knn.predict(X_test)

#Results analysis
model_labels=['SVM','NB','KNN']
accuracy= [round(clf_svm.score(X_test,y_test)*100), round(clf_nb.score(X_test,y_test)*100), round(clf_knn.score(X_test,y_test)*100)]

axis=sns.barplot(x=model_labels, y=accuracy, color='blue')
for i in axis.patches:
    axis.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2, i.get_height()),
                  ha='center', va='baseline',fontsize='13',
                  xytext=(0,1),textcoords='offset points')

plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy in First Stage')
plt.ylim((0, 80))
plt.show()


svm_proba=clf_svm.predict_proba(X_test)
nb_proba=clf_nb.predict_proba(X_test)
knn_proba=clf_knn.predict_proba(X_test)
r_proba=[0 for i in range(len(y_test))]

svm_auc_score=roc_auc_score(y_test,svm_proba, multi_class='ovr')
nb_auc_score=roc_auc_score(y_test,nb_proba, multi_class='ovr')
knn_auc_score=roc_auc_score(y_test,knn_proba, multi_class='ovr')

# Alegria

plt.figure(figsize=(10,7))

fpr, tpr, _ = roc_curve(y_test,r_proba, pos_label=0)

fpr_svm, tpr_svm, _ = roc_curve(y_test,svm_proba[:,0], pos_label=0)
fpr_nb, tpr_nb, _ = roc_curve(y_test,nb_proba[:,0], pos_label=0)
fpr_knn, tpr_knn, _ = roc_curve(y_test,knn_proba[:,0], pos_label=0)

plt.plot(fpr_svm, tpr_svm, linestyle='--', color='red', label='Support Vector Machine')
plt.plot(fpr_nb, tpr_nb, linestyle='--', color='green', label='Naive Bayes')
plt.plot(fpr_knn, tpr_knn, linestyle='--', color='purple', label='K Nearest Neighbors')
plt.plot(fpr, tpr, linestyle='solid', color='blue')

plt.title('Curva ROC')
plt.xlabel('True Rate Positive - Alegria')
plt.ylabel('False Rate Positive - Alegria')
plt.legend()
plt.show()
# Surpresa

fpr, tpr, _ = roc_curve(y_test,r_proba, pos_label=1)

fpr_svm, tpr_svm, _ = roc_curve(y_test,svm_proba[:,1], pos_label=1)
fpr_nb, tpr_nb, _ = roc_curve(y_test,nb_proba[:,1], pos_label=1)
fpr_knn, tpr_knn, _ = roc_curve(y_test,knn_proba[:,1], pos_label=1)

plt.figure(figsize=(10,7))
plt.plot(fpr_svm, tpr_svm, linestyle='--', color='red', label='Support Vector Machine')
plt.plot(fpr_nb, tpr_nb, linestyle='--', color='green', label='Naive Bayes')
plt.plot(fpr_knn, tpr_knn, linestyle='--', color='purple', label='K Nearest Neighbors')
plt.plot(fpr, tpr, linestyle='solid', color='blue')

plt.title('Curva ROC')
plt.xlabel('True Rate Positive - Surpresa')
plt.ylabel('False Rate Positive - Surpresa')
plt.legend()
plt.show()

# Tristeza

fpr, tpr, _ = roc_curve(y_test,r_proba, pos_label=2)

fpr_svm, tpr_svm, _ = roc_curve(y_test,svm_proba[:,2], pos_label=2)
fpr_nb, tpr_nb, _ = roc_curve(y_test,nb_proba[:,2], pos_label=2)
fpr_knn, tpr_knn, _ = roc_curve(y_test,knn_proba[:,2], pos_label=2)

plt.figure(figsize=(10,7))
plt.plot(fpr_svm, tpr_svm, linestyle='--', color='red', label='Support Vector Machine')
plt.plot(fpr_nb, tpr_nb, linestyle='--', color='green', label='Naive Bayes')
plt.plot(fpr_knn, tpr_knn, linestyle='--', color='purple', label='K Nearest Neighbors')
plt.plot(fpr, tpr, linestyle='solid', color='blue')
plt.title('Curva ROC')
plt.xlabel('True Rate Positive - Tristeza')
plt.ylabel('False Rate Positive - Tristeza')
plt.legend()
plt.show()