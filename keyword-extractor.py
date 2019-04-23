# keyword-extract.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, requests, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class ExtractKeyWords():
    
    '''
    This class takes in a raw text, source name, and brief description then:
    1) removes punctuations from raw text
    2) returns a list of all separated words
    3) reduces the full list of all words to only key words (no stopwords)
    4) lemmatizes the key words for relevance
    5) creates a paragraph of keywords for wordclouds
    '''
    
    def __init__(self, raw_text, source_name, description):
        
        self.raw_text = raw_text
        self.source_name = source_name
        self.description = description
        self.paragraph = ''
        self.allwords = []
        self.kwords = []
        self.joined_kwords = ''
        self.processed_words = []
    
    def nopunc(self):
        '''
		Using re was more effective to removing punctuation than filtering via string.punctuation.
        '''
        self.paragraph = self.raw_text.read()
        self.allwords = re.findall('[^!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\s ]+',self.paragraph)
        #return self.allwords
    
    def keywords(self):
        self.kwords = (word.lower().strip() for word in self.allwords)
        self.kwords = [word for word in self.kwords if word == 'r' or word not in stopwords.words('english') and len(word) > 2]
        #return self.kwords
    
    def lemmatizer(self):
          
        wordnet_lemmatizer = WordNetLemmatizer()

        for word in self.kwords:
            syn = wordnet_lemmatizer.lemmatize(word)
            self.processed_words.append(syn)     
        #return self.processed_words

    def join_kwords(self):
        self.joined_kwords = ' '.join(self.kwords)
        #return self.joined_kwords
        

def syn_word_clm(df,clm_name):
    for i in range(0,len(df)):
        try:
            yield wordnet.synsets(df.loc[i,clm_name])[0].name()
        
        except IndexError:
            yield None

def word_type(row):
    if row == None:
        return None
    else:
        return row[-5:-2]

def score(col1,col2):
    
    for i in range(0,len(col1)):
        try:
            w1 = wordnet.synset(col1.loc[i])
            w2 = wordnet.synset(col2.loc[i]) 
            yield w1.wup_similarity(w2)
    
        except:
            yield None


if __name__ == '__main__':

	doc = open('your_file.txt','r')
	wp = ExtractKeyWords(raw_text=doc, source_name='where did you get your document', description='description, keywords, or tags')

	wp.nopunc()
	wp.keywords()
	wp.join_kwords()
	wp.lemmatizer()

	d = {'Words': wp.processed_words}
	df = pd.DataFrame(data=d)
	x = df['Words'].value_counts()
	dfx = pd.DataFrame(x)
	top = dfx[(dfx.index == 'r') | (dfx['Words'] > 2)]
	top.reset_index(inplace=True)
	top.columns = ['Word','Count']

	top = top.assign(Synonym = list(syn_word_clm(top,'Word')))
	top = top.assign(Word_Eval = top['Word'] + top['Synonym'].apply(word_type) + '01')
	top = top.assign(Similarity = list(score(top['Word_Eval'],top['Synonym'])))

	print(f'Source:      {wp.source_name.capitalize()}')
	print(f'Description: {wp.description.capitalize()}')
	print(f'\n{top.head(10)}')

	sns.set_style('whitegrid')
	sns.set_context("paper",font_scale=1.2)
	f, ax = plt.subplots(figsize=(12, 16), dpi = 120)

	g = sns.barplot(x='Count', y='Word', data=top, palette='Blues_d', label='Key Words')
	g2 = sns.barplot(x=top['Count'].max(), y='Word', data=top, color='r', label='Max')
	scale = list(range(0,top['Count'].max()+2,2))
	plt.setp(ax,xticks=scale)
	plt.show()