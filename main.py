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

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()

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


class UploadCsv(Resource):
    def post(self):
        # get file csv
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parser.parse_args()
        file_csv = args['file']
        file_csv.save("files/your_file_name.csv")
        df=pd.read_csv("files/your_file_name.csv")

        df['clean']=df['review'].str.replace(r"(^| ).(( ).)*( |$)"," ")

        clean_materi = df['clean'].apply(lambda x: clean_text(x))

        token_materi = clean_materi.apply(df_tokenizer)

        materi_nostopword = token_materi.apply(stopword_removal)

        materi_ok = materi_nostopword.apply(sentencizer)
        df['text_prepocessing'] = materi_ok

        return {
          "data_csv": {
            "review": convert_csv_to_array(df.to_numpy(),0),
            "clean": convert_csv_to_array(df.to_numpy(),1),
            "text_prepocessing": convert_csv_to_array(df.to_numpy(),2)
          }
        }

api.add_resource(UploadCsv, '/api/v1/upload-file-csv')

if __name__ == '__main__':
    app.run(debug=True)