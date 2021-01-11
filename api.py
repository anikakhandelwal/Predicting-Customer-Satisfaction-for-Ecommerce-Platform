#!/usr/bin/env python
# coding: utf-8

# In[13]:


from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from scipy.sparse import hstack,coo_matrix,vstack
import pickle
import re
import datetime
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class Label_Encoder():
    '''
    In this class, we have created a label encoder which takes any categorical feature and encode it 
    by giving labels from 0 to n where n is the total number of unique values in the features.
    '''
    def __init__(self):
        '''
        This method is called whenever any object is created for this class. It initializes an empty dictionary
        to store the vocab for that feature with the values as labels.
        '''
        self.vocab = dict()
    
    def fit(self,data):
        '''
        This is the fit method of the label encoder. It extracts the vocab from the feature and stores it in the 
        dictionary along with an additional key called 'unseen'. This key is used to handle if any previously
        unseen value is encountered at the time of transforming the feature.
        '''
        self.classes = np.unique(list(data))
        self.vocab = { self.classes[i]:i+1 for i in range(0,len(self.classes))}
        self.vocab['unseen'] = 0
        
    def transform(self,data):
        '''
        This is the transform function of the label encoder. It takes the data to tansform as an argument. For 
        every value, it looks up in the dictionary to find a match and replace it with its label. If a match
        is not found it relaces it with the label for the unseen values.
        '''
        data = [i if i in self.vocab.keys() else 'unseen' for i in list(data)]
        transformed_data = [self.vocab[value] for value in data]
        return np.array(transformed_data).reshape(-1,1)
        
#datafiles = "/media/anika/DATA/Documents/Python_Scripts/Self_Case_Study/CS1/preprocess_files/"
standard_price = pickle.load(open("price_standardizer.pkl","rb"))
standard_freight = pickle.load(open("freight_value_standardizer.pkl","rb"))
standard_payment= pickle.load(open("payment_value_standardizer.pkl","rb"))
standard_shipping = pickle.load(open("shipping_days_standardizer.pkl","rb"))
standard_length = pickle.load(open("review_length_standardizer.pkl","rb"))
le = pickle.load(open("seller_zip_code_prefix_le.pkl","rb"))
clf = pickle.load(open("xgb_stan.pkl","rb"))
w2v_model = pickle.load(open("self_trained_w2v.pkl","rb"))

def is_review_verified(d):
    if(d==False):
        return 1
    else:
        return 0
    
def convert_to_datetime(d):
    '''
    This function converts the string object to a datetime object.
    '''
    try:
        d = datetime.datetime.strptime(d, '%Y-%m-%d').date()
    except:
        d = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date()    
    return d    

def convert_to_days(d):
    '''
    This functions returns the number of days from the days object of the class datetime.
    '''
    return d.days

def is_early_or_late(num):
    '''
    This function checks the sign of the input. If it is less than 0 then it returns 0 else it returns 1.
    '''
    if (num < 0):
        return 0
    else:
        return 1

def calc_review_length(r):
    '''
    This function takes a review comment message as argument and returns the length of the review.
    '''
    return len(r.split(" "))

def review_has_number(r):
    '''
    This function checks whether the review comment message has any number or not. If a number is present, it 
    returns 1 else 0 is returned. 
    '''
    r = r.split(" ")
    r = int(any(ch.isdigit() for ch in r))
    return r

def review_has_negative_word(r):
    '''
    This function takes review comment message as an argument. It checks whether a negation word is present in 
    the review or not. If it finds a word, it returns 1 else 0.
    '''
    negation_words = {'nada','ninguém','nenhum','nenhuma','tampouco','nem','nunca',
                  'jamais','não','negativa','negativo','adulterada', 'adulterado',
                 'usava', 'partida','partido','errada','errado'}
    r = r.split(" ")
    r = int(any(ch in negation_words for ch in r))
    return r

def func_null_review(review):
    '''
    This function checks whether a review comment is present or not. If it is not present, it returns 'sem revisão'
    '''
    if (type(review)!= str):
        return 'sem revisão'
    else:
        return review
    
    
def text_preprocessing(reviews):
    '''
    This function takes a series of review comment messages and processes them one by one. It returns a list of
    cleaned review messages.
    '''
   
    ptext = []
    for text in list(reviews):
        text=re.sub("[<].*[>]","",text)            #Removing tags
        text=re.sub("[(].*[)]","",text)            #Removing brackets
        text=text.replace("\n"," ")                #Removing \n
        text=text.replace("\t"," ")                #Removing \t
        text=text.replace("-"," ")                 #Removing -
        text=text.replace("\\","")                 #Removing \
        text=text.replace("!","")                  #Removing !
        text=text.replace("."," ")                 #Removing .
        text=text.replace("/"," ")                 #Removing /
        text=text.replace("R$","")                 #Removing Currency symbol
        text=re.sub("[0-9]","",text)               #Removing numbers
        text = str(text).lower()                   #Converting to lowercase
        text=re.sub(r'\b\w{1,2}\b',"",text)        #Removing small words
        text=re.sub(r'\b\w{15,}\b',"",text)        #Removing large words
        ptext.append(text)
   
    return ptext

def preprocessing_without_text_standardization(data):
    '''
    This function is used to preprocess the features other than the text feature.
    '''
    #price
    data_price = standard_price.transform(data["price"].values.reshape(-1,1)).reshape(-1,)
    
    #freight_value
    data_freight_value = standard_freight.transform(data["freight_value"].values.reshape(-1,1)).reshape(-1,)
    
    #payment_value
    data_payment_value = standard_payment.transform(data["payment_value"].values.reshape(-1,1)).reshape(-1,)

    #shipping_days
    data_shipping_days = standard_shipping.transform(data["shipping_days"].values.reshape(-1,1)).reshape(-1,)
    
    #review_length
    data_review_length = standard_length.transform(data["review_length"].values.reshape(-1,1)).reshape(-1,)

    #seller_zip_code_prefix
    data_seller_zip_code_prefix = le.transform(data["seller_zip_code_prefix"].values).reshape(-1,)   
    
    data_preprocessed = vstack((data_price,data_freight_value,data_payment_value,data_shipping_days,
                               data_review_length,data_seller_zip_code_prefix,coo_matrix(data["review_verified"].to_numpy()),
                               coo_matrix(data["early_or_late"].to_numpy()),coo_matrix(data["has_num"].to_numpy()), 
                               coo_matrix(data["has_negative_word"].to_numpy()))).tocsr()
    
    return data_preprocessed.transpose()

def w2v_vectorizer(text,model):
    '''
    This function takes up a list of preprocessed data and a trained word2vec model and returns the 
    average word2vec vector for each sentence.
    '''
    words = list(model.wv.vocab)
    text_w2v = []
    for rev in text:
        vector = np.zeros(300)
        c = 0
        for word in rev:
            if(word in words):
                vector += model[word]
                c += 1
        if (c>0):
            vector /= c
        text_w2v.append(vector)
    return text_w2v

@app.route('/', methods=["GET"])
def start():
    return render_template('index.html')

@app.route('/upload', methods = ["POST"])
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ["POST"])
def uploaded_file():
    f = request.files['file']
    x=pd.read_csv(f)
    x["review_verified"] = x["order_delivered_customer_date"].isnull().apply(is_review_verified)
    x['order_delivered_customer_date'] = np.where(x['order_delivered_customer_date']==' ',
                                                  x['order_estimated_delivery_date'], x['order_delivered_customer_date'])
    x["order_purchase_timestamp"] = x["order_purchase_timestamp"].apply(convert_to_datetime)
    x["order_delivered_customer_date"] = x["order_delivered_customer_date"].apply(convert_to_datetime)
    x["order_estimated_delivery_date"] = x["order_estimated_delivery_date"].apply(convert_to_datetime)
    x["shipping_days"] = x["order_delivered_customer_date"] - x["order_purchase_timestamp"]
    x["shipping_days"] = x["shipping_days"].apply(convert_to_days)
    x["eta_in_days"] = x["order_estimated_delivery_date"] - x["order_purchase_timestamp"]
    x["eta_in_days"] = x["eta_in_days"].apply(convert_to_days)
    x["early_or_late"] = x["eta_in_days"] - x["shipping_days"]
    x["early_or_late"] = x["early_or_late"].apply(is_early_or_late)
    x = x.drop(['order_delivered_customer_date','order_estimated_delivery_date',
              'order_purchase_timestamp','eta_in_days'],axis = 1)
    x["review_comment_message"] = x["review_comment_message"].apply(func_null_review)
    x["review_length"] = x["review_comment_message"].apply(calc_review_length)
    x["has_num"] = x["review_comment_message"].apply(review_has_number)
    x["has_negative_word"] = x["review_comment_message"].apply(review_has_negative_word)
    data_standardized = preprocessing_without_text_standardization(x)
    text_data_preprocessed = text_preprocessing(x["review_comment_message"])
    #w2v
    text_w2v = w2v_vectorizer(text_data_preprocessed,w2v_model)
    final_data = hstack((data_standardized,text_w2v)).tocsr()
    #model predict
    y_pred = clf.predict(final_data)
    #y_pred=y_pred.tolist()
    y_pred=['Positive' if i==1 else 'Negative' for i in y_pred]
    return jsonify({'Predicted Values': y_pred})

@app.route('/predict', methods=['POST'])
def function1():
    data = request.form.to_dict()
    data = pd.DataFrame([data.values()], columns=data.keys())
    x = data[["price","freight_value","payment_value","seller_zip_code_prefix",
            "review_comment_message","order_delivered_customer_date",
            "order_purchase_timestamp","order_estimated_delivery_date"]]
    x["review_verified"] = x["order_delivered_customer_date"].isnull().apply(is_review_verified)
    print(x['order_delivered_customer_date'].iloc[0])
    x['order_delivered_customer_date'] = np.where(x['order_delivered_customer_date']=='',
                                                  x['order_estimated_delivery_date'], x['order_delivered_customer_date'])
    
    x["order_purchase_timestamp"] = x["order_purchase_timestamp"].apply(convert_to_datetime)
    x["order_delivered_customer_date"] = x["order_delivered_customer_date"].apply(convert_to_datetime)
    x["order_estimated_delivery_date"] = x["order_estimated_delivery_date"].apply(convert_to_datetime)
    x["shipping_days"] = x["order_delivered_customer_date"] - x["order_purchase_timestamp"]
    x["shipping_days"] = x["shipping_days"].apply(convert_to_days)
    x["eta_in_days"] = x["order_estimated_delivery_date"] - x["order_purchase_timestamp"]
    x["eta_in_days"] = x["eta_in_days"].apply(convert_to_days)
    x["early_or_late"] = x["eta_in_days"] - x["shipping_days"]
    x["early_or_late"] = x["early_or_late"].apply(is_early_or_late)
    x = x.drop(['order_delivered_customer_date','order_estimated_delivery_date',
              'order_purchase_timestamp','eta_in_days'],axis = 1)
    x["review_comment_message"] = x["review_comment_message"].apply(func_null_review)
    x["review_length"] = x["review_comment_message"].apply(calc_review_length)
    x["has_num"] = x["review_comment_message"].apply(review_has_number)
    x["has_negative_word"] = x["review_comment_message"].apply(review_has_negative_word)
    data_standardized = preprocessing_without_text_standardization(x)
    text_data_preprocessed = text_preprocessing(x["review_comment_message"])
    #w2v
    text_w2v = w2v_vectorizer(text_data_preprocessed,w2v_model)
    final_data = hstack((data_standardized,text_w2v)).tocsr()
    #model predict
    y_pred = clf.predict(final_data)
    if(y_pred==1):
        y_pred = 'Positive'
    else:
        y_pred = 'Negative'
    return jsonify({'Predicted Values': y_pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

