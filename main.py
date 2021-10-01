from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import werkzeug
import re
import spacy
from spacy.lang.id import Indonesian
import nltk
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
import numpy as np
import csv

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
lexicon = dict()

def convert_csv_to_array(df_numpy,column):
    result = []
    for item in df_numpy:
        result.append(item[column])
    return result

def clean_text(Text):
    # remove backslash-apostrophe 
    Text = re.sub("\'", "", Text) 
    # remove everything except alphabets 
    Text = re.sub("[^a-zA-Z]"," ",Text) 
    # remove whitespaces 
    Text = ' '.join(Text.split()) 
    # convert text to lowercase 
    Text = Text.lower() 

    return Text

def df_tokenizer(df):
    nlp = Indonesian()  # use directly
    print(df)
    tokenized = [token.text for token in nlp(df)]
    return tokenized

def stopword_removal(token):
    stopwords = spacy.lang.id.stop_words.STOP_WORDS
    stopwords.add("yuk")
    stopwords.add("ya")
    stopwords.add("tuh")
    stopwords.add("sih")
    stopwords.add("ngga")
    stopwords.add("nggak")
    stopwords.add("yak")
    stopwords.add("lho")
    stopwords.add("loh")
    stopwords.add("yak")
    stopwords.add("deh")

    clean = []
    for i in token:
        if i not in stopwords:
            clean.append(i)
    return clean

def sentencizer(token):
    value = " ".join(str(v) for v in token) #join as string
    return value

# berfungsi untuk menulis sentimen kata
def found_word(ind,words,word,sen,sencol,sentiment,add):
    lexicon = pd.read_csv('static_file/modified_full_lexicon.csv')
    lexicon = lexicon.reset_index(drop=True)

    negasi = ['bukan','tidak','ga','gk']
    lexicon_word = lexicon['word'].to_list()
    # jika sudah termasuk dalam bag of words matrix, maka tinggal menambah nilainya
    if word in sencol:
        sen[sencol.index(word)] += 1
    else:
    #jika tidak, menambahkan kata baru
        sencol.append(word)
        sen.append(1)
        add += 1
    if (words[ind-1] in negasi):
        sentiment += -lexicon['weight'][lexicon_word.index(word)]
    else:
        sentiment += lexicon['weight'][lexicon_word.index(word)]

    return sen,sencol,sentiment,add


class UploadCsv(Resource):
    def post(self):
        # get file csv
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parser.parse_args()
        file_csv = args['file']
        file_csv.save("files/your_file_name.csv")
        df=pd.read_csv("files/your_file_name.csv")


        # preprocessing
        df['clean']=df['review'].str.replace(r"(^| ).(( ).)*( |$)"," ")

        clean_materi = df['clean'].apply(lambda x: clean_text(x))

        token_materi = clean_materi.apply(df_tokenizer)

        materi_nostopword = token_materi.apply(stopword_removal)

        materi_ok = materi_nostopword.apply(sentencizer)
        df['text_preprocessing'] = materi_ok

        data_text = []
        for index,item in enumerate(df['review']):
            data_text.append({
                "review": item,
                "clean": convert_csv_to_array(df.to_numpy(),1)[index],
                "text_preprocessing": convert_csv_to_array(df.to_numpy(),2)[index]
            })


        # pelabelan
        pos_lexicon = pd.read_csv('static_file/positive - positive.csv',sep='\t')
        neg_lexicon = pd.read_csv('static_file/negative - negative.csv',sep='\t')

        lexicon = pd.read_csv('static_file/modified_full_lexicon.csv')
        lexicon = lexicon.reset_index(drop=True)

        negasi = ['bukan','tidak','ga','gk']
        lexicon_word = lexicon['word'].to_list()
        lexicon_num_words = lexicon['number_of_words']

        sencol =[]
        senrow =np.array([])
        nsen = 0
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        sentiment_list = []

        # memeriksa setiap kata, jika mereka muncul dalam leksikon, dan kemudian menghitung sentimen mereka jika mereka muncul
        for i in range(len(df)):
            nsen = senrow.shape[0]
            words = word_tokenize(df["text_preprocessing"][i])
            sentiment = 0 
            add = 0
            prev = [0 for ii in range(len(words))]
            n_words = len(words)
            if len(sencol)>0:
                sen =[0 for j in range(len(sencol))]
            else:
                sen =[]
            
            for word in words:
                ind = words.index(word)
                # periksa apakah mereka termasuk dalam leksikon
                if word in lexicon_word :
                    sen,sencol,sentiment,add= found_word(ind,words,word,sen,sencol,sentiment,add)
                else:
                # if not, then check the root word
                    kata_dasar = stemmer.stem(word)
                    if kata_dasar in lexicon_word:
                        sen,sencol,sentiment,add= found_word(ind,words,kata_dasar,sen,sencol,sentiment,add)
                # jika masih negatif, coba cocokkan kombinasi kata dengan kata yang berdekatan
                    elif(n_words>1):
                        if ind-1>-1:
                            back_1    = words[ind-1]+' '+word
                            if (back_1 in lexicon_word):
                                sen,sencol,sentiment,add= found_word(ind,words,back_1,sen,sencol,sentiment,add)
                            elif(ind-2>-1):
                                back_2    = words[ind-2]+' '+back_1
                                if back_2 in lexicon_word:
                                    sen,sencol,sentiment,add= found_word(ind,words,back_2,sen,sencol,sentiment,add)
            if add>0:  
                if i>0:
                    if (nsen==0):
                        senrow = np.zeros([i,add],dtype=int)
                    elif(i!=nsen):
                        padding_h = np.zeros([nsen,add],dtype=int)
                        senrow = np.hstack((senrow,padding_h))
                        padding_v = np.zeros([(i-nsen),senrow.shape[1]],dtype=int)
                        senrow = np.vstack((senrow,padding_v))
                    else:
                        padding =np.zeros([nsen,add],dtype=int)
                        senrow = np.hstack((senrow,padding))
                    senrow = np.vstack((senrow,sen))
                if i==0:
                    senrow = np.array(sen).reshape(1,len(sen))
            # jika tidak ada maka perbarui saja matriks lama
            elif(nsen>0):
                senrow = np.vstack((senrow,sen))
                
            sentiment_list.append(sentiment)

        sencol.append('sentiment')
        sentiment_array = np.array(sentiment_list).reshape(senrow.shape[0],1)
        sentiment_data = np.hstack((senrow,sentiment_array))
        df_sen = pd.DataFrame(sentiment_data,columns = sencol)

        cek_df = pd.DataFrame([])
        cek_df['text'] = df["text_preprocessing"].copy()
        cek_df['sentiment']  = df_sen['sentiment'].copy()

        data_label = []
        global positive
        global negative
        positive = 0
        negative = 0

        def getLabelHasil(value):
            global positive
            global negative
            if value >= 0:
                positive+= 1
                return "positif"
            else:
                negative+= 1
                return "negatif"

        for index,item in enumerate(cek_df['text']):
            
            data_label.append({
                "text": item,
                "sentiment": convert_csv_to_array(cek_df.to_numpy(),1)[index],
                "hasil": getLabelHasil(convert_csv_to_array(cek_df.to_numpy(),1)[index])
            })

        return {
            "data_text": {
                "list": data_text
            },
            "data_label": {
                "list": data_label,
                "percent": {
                    "positive": positive/len(data_label),
                    "negative": negative/len(data_label)
                }
            }
        }

api.add_resource(UploadCsv, '/api/v1/upload-file-csv')

if __name__ == '__main__':
    app.run(debug=True)