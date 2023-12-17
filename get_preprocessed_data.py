from utils import preprocess
import pandas as pd
if __name__ == '__main__':

    obj = preprocess()
    obj.get_text(url = "data\\testdata.manual.2009.06.14.csv",drop=False,save=True)
    sentence1 = "Ok, first assesment of the #kindle2 ...it fucking rocks!!!"
    sentence2 = "Fuck this economy. I hate aig and their non loan given asses."
    data = {'label': ['4','0'],
            'contents': [sentence1,sentence2]}

    df = pd.DataFrame(data)
    obj.testandtry(df['contents'])