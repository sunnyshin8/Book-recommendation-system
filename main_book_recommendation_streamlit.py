import streamlit as st
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import warnings
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')




# load model
@st.cache_data
def main():
    #load data
    df = pd.read_csv(r'book_final.csv')
    data_k = pd.read_csv(r'data_K.csv')
    from sklearn.neighbors import NearestNeighbors
    model1 = NearestNeighbors(metric = 'euclidean', algorithm = 'ball_tree') 
    # model1 = pickle.load(open('model1.pkl','rb'))    
    model1.fit(data_k)


    distances, indices = model1.kneighbors(data_k[['encoded_language', 'average_rating', 'ratings_count', 'text_reviews_count']], n_neighbors = 6)  


    for i in range(len(indices)):
        book_index = df.index[indices[i]] 
        print(df.iloc[book_index, :]) 


    final_data = df.copy()
    final_data['indices'] = indices.tolist()
    final_data['distances'] = distances.tolist()
    return df , final_data.indices


import re
class BookQuest:
    def __init__(self, dataframe, indices):
        self.df = dataframe
        self.indices = indices
        self.all_books_names = list(self.df["title"].values)

    def find_id(self,name):
        for index,string in enumerate(self.all_books_names):
            if re.search(name,string):
                index=index
                break
        return(index)

    def print_similar_books(self, query=None):
        if query:
            found_id = self.find_id(query)
            ret_books = []
            for id in self.indices[found_id][1:]:
                # print(id)
                ret_books.append(self.df.iloc[id]["title"])
            return ret_books
    
## web page

st.set_page_config(page_title='Book Recommendation System', page_icon='📚', layout='wide', initial_sidebar_state='auto')
st.title('Book Recommendation System')
st.spinner('Loading...')

df , indices = main()

# Query text
query_text = st.text_input('Enter book name :-',label_visibility="visible", disabled=False, max_chars=None, key=None, type='default')


# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not( query_text))
    # After Submission
    if submitted:
        with st.spinner('Calculating...'):
            recsys = BookQuest(df,indices)
            response = recsys.print_similar_books(query_text)
            result.append(response)

if len(result):
    st.info('Recommendations for the entered book is shown here')
    for i in range(1,len(response)+1):
        st.info(f'{i}. {response[i-1]}')