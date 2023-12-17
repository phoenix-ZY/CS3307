import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import enchant
from nltk.corpus import wordnet
from nltk import pos_tag
import pandas as pd
from tqdm import tqdm
import time
# nltk.download()


def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    return accuracy
def easy_preprocess(data):
    X = data['tweets'].tolist()
    y_dict = {0: 0,  4: 1}
    y = data['label'].map(y_dict).tolist()
    return X,y

class preprocess:
    _r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    _url_pattern = re.compile(r'https?://\S+|www\.\S+')
    _at_pattern = re.compile(r'@[^ ]+')
    _USdict = enchant.Dict("en_US")
    def _remove_notes(self,text):
        text = re.sub(self._at_pattern, ' ', text)
        text = re.sub(self._url_pattern, ' ', text)
        text = re.sub(self._r4, '', text)
        # print(text)
        return text

    def _remove_stopwords(self,text):
        removedtext = [w for w in text.split(' ') if w not in stopwords.words('english')]
        elements_to_remove = ' '
        removedtext = [item for item in removedtext if item not in elements_to_remove]
        # removedtext = ' '.join(removedtext)
        # print(removedtext)
        return removedtext

    def _lemmatization(self,text):
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
        text = [word.lower() for word in text if self._USdict.check(word) and not str.isdigit(word) and len(word) > 2]
        text = ' '.join(text)
        # print(text)
        return text

    def get_text(self,url,drop=True,samplenum=10000,save=False):
        train_file = pd.read_csv(url, encoding='ISO-8859-1',
                                 header=None,
                                 names=['label', 'id', 'day', 'query', 'user', 'tweets'])  # pandas dataframe自定义列表名
        train_file = train_file.drop(train_file.columns[1:5], axis=1)
        # train_file = train_file.sample(frac=1).reset_index(drop=True)
        if drop:
            train_file = train_file.head(samplenum)
        tqdm.pandas()
        train_file['tweets'] = train_file['tweets'].progress_apply(self._remove_notes)

        train_file['tweets'] = train_file['tweets'].progress_apply(self._remove_stopwords)

        train_file['tweets'] = train_file['tweets'].progress_apply(self._lemmatization)

        train_file = train_file[~(train_file['tweets'].str.len() == 0)]

        if save:
            train_file.to_csv('data/test_processed.csv', encoding='utf-8', index=False)
        return train_file

    def testandtry(self,text):
        text = [self._remove_notes(sen) for sen in text]
        text = [self._remove_stopwords(sen) for sen in text]
        text = [self._lemmatization(sen) for sen in text]
        print(text)
        return text