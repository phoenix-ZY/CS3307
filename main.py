import enchant
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import TweetTokenizer
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
url_pattern = re.compile(r'https?://\S+|www\.\S+')
at_pattern = re.compile(r'@[^ ]+')

def remove_notes(text):
    text = re.sub(at_pattern, ' ', text)
    text = re.sub(url_pattern, ' ', text)
    text = re.sub(r4,'',text)
    #print(text)
    return text

#去除停用词
def remove_stopwords(text):
    removedtext = [w for w in text.split(' ') if w not in stopwords.words('english')]
    elements_to_remove = ' '
    removedtext = [item for item in removedtext if item not in elements_to_remove]
    # removedtext = ' '.join(removedtext)
    #print(removedtext)
    return removedtext

#词性还原
USdict = enchant.Dict("en_US")

def lemmatization(text):
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    text = pos_tag(text)
    wnl = WordNetLemmatizer()
    text = [wnl.lemmatize(tag[0], pos=get_wordnet_pos(tag[1]) or wordnet.NOUN) for tag in text]
    text = [word.lower() for word in text if not str.isdigit(word) and len(word) > 2]
    text = ' '.join(text)
    #print(text)
    return text


if __name__ == '__main__':
    # train_file = pd.read_csv('trainingandtestdata/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1',
    #                    header=None, names=['label', 'id', 'day', 'query', 'user', 'tweets'])  # pandas dataframe自定义列表名
    test_file = pd.read_csv('trainingandtestdata/testdata.manual.2009.06.14.csv', encoding='ISO-8859-1',
                       header=None, names=['label', 'id', 'day', 'query', 'user', 'tweets'])
    # #train_file = train_file.drop(train_file.columns[1:5], axis=1)  # 删除多余列
    test_file = test_file.drop(test_file.columns[1:5], axis=1)  # 删除多余列
    # # shuffle it
    # train_file = train_file.sample(frac=1).reset_index(drop=True)
    # #get first 5000
    #train_file = train_file.head(60000)
    # #visualize progress
    # tqdm.pandas()
    # train_file['tweets'] = train_file['tweets'].progress_apply(remove_notes)
    test_file['tweets'] = test_file['tweets'].apply(remove_notes)
    #
    # # train_file['tweets'] = train_file['tweets'].progress_apply(remove_stopwords)
    # # # # test_file['tweets'] = test_file['tweets'].apply(remove_stopwords)
    # # #
    # # train_file['tweets'] = train_file['tweets'].progress_apply(lemmatization)
    # # # test_file['tweets'] = test_file['tweets'].apply(lemmatization)
    #
    # # 保存文本
    # train_file.to_csv('Test5000.csv', encoding='ISO-8859-1',index = False)
    #
    train_file = pd.read_csv('Test5000.csv',encoding='ISO-8859-1')
    train_file = train_file.dropna(subset=['tweets'])
    train_file = train_file[~(train_file['tweets'].str.len() == 0)]
    train_file = train_file.head(160000)
    #词袋模型
    bow_vectorizer = CountVectorizer(max_df=0.80, min_df=2, max_features=10000)


    # TF-IDF feature
    tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=2, max_features=10000)

    x = train_file['tweets']
    y = train_file['label']
    # print(x[1:10])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn import metrics

    # KNN Classifier K近邻算法
    Knn = KNeighborsClassifier()

    # Logistic Regression Classifier 逻辑回归
    Lr = LogisticRegression()

    # Random Forest Classifier 随机森林
    Rf = RandomForestClassifier(
    n_estimators=1000,       # Number of trees in the forest
    criterion='entropy',       # Criterion for splitting ('gini' or 'entropy')
    max_depth=None,          # Maximum depth of the tree
    min_samples_split=10,     # Minimum number of samples required to split an internal node
    min_samples_leaf=1,      # Minimum number of samples required to be at a leaf node
    max_features='sqrt',    # Number of features to consider for the best split ( or 'sqrt' or 'log2' or None)
    n_jobs=-1,              # Number of jobs to run in parallel (-1 uses all available cores)
    random_state=42         # Seed for random number generation
    )

    # SVM Classifier 支持向量机
    Svm = SGDClassifier()

    # Naive Bayes 朴素贝叶斯
    Nb = MultinomialNB()

    pipe = make_pipeline(bow_vectorizer, Rf)
    pipe.fit(x_train, y_train)
    # y_pred = pipe.predict(x_test)  # 进行预测
    # df = pd.DataFrame()
    # df['pred'] = y_pred  # 将预测结果保存到dataframe中
    # print(metrics.classification_report(y_test, y_pred))
    #
    y_test = test_file['label']
    y_pred = pipe.predict(test_file['tweets'])
    print(metrics.classification_report(y_test, y_pred))