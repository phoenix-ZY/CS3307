import preprocess
import pandas as pd
if __name__ == '__main__':

    obj = preprocess.preprocess()
    url = "trainingandtestdata/training.1600000.processed.noemoticon.csv"
    preprocess.preprocess.get_text(obj,url)
    sentence1 = "Ok, first assesment of the #kindle2 ...it fucking rocks!!!"
    sentence2 = "Fuck this economy. I hate aig and their non loan given asses."
    data = {'label': ['4','0'],
            'contents': [sentence1,sentence2]}

    df = pd.DataFrame(data)
    preprocess.preprocess.testandtry(obj,df['contents'])
