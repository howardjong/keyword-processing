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
		print("Raw paragraph:", self.paragraph)  # Debug: show file content
		# Try a simpler regex to extract words
		self.allwords = re.findall(r'\w+', self.paragraph)
		print("Extracted words with r'\\w+':", self.allwords)  # Debug: show extracted words
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
        except IndexError as e:
            print(f"IndexError in syn_word_clm at row {i} for value '{df.loc[i,clm_name]}': {e}")
            yield None
        except Exception as e:
            print(f"Error in syn_word_clm at row {i} for value '{df.loc[i,clm_name]}': {e}")
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
        except Exception as e:
            print(f"Error in score at index {i} with values ({col1.loc[i]}, {col2.loc[i]}): {e}")
            yield None


if __name__ == '__main__':

    doc = open('analyze-text.txt','r')
    wp = ExtractKeyWords(raw_text=doc, source_name='where did you get your document', description='description, keywords, or tags')

    wp.nopunc()
    print("All words after nopunc:", wp.allwords)
    wp.keywords()
    print("Keywords after filtering:", wp.kwords)
    wp.lemmatizer()
    wp.join_kwords()
    print("Lemmatized words:", wp.processed_words)

    d = {'Words': wp.processed_words}
    df = pd.DataFrame(data=d)
    print("DataFrame of processed words:\n", df)
    x = df['Words'].value_counts()
    dfx = pd.DataFrame(x)
    dfx.columns = ['Count']  # Rename the count column
    dfx.index.name = 'Word' # Name the index for clarity
    print("Value counts before filtering:")
    print(dfx)

    top = dfx  # Show all words, no filtering

    # Check for empty DataFrame before further processing
    if top.empty:
        print("No keywords available after filtering. Skipping visualization.")
    else:
        top = top.reset_index()  # Now columns are ['Word', 'Count']
        top = top.assign(Synonym = list(syn_word_clm(top,'Word')))
        top['Word_Eval'] = top.apply(
            lambda row: str(row['Word']) + (str(word_type(row['Synonym'])) if word_type(row['Synonym']) is not None else '') + '01',
            axis=1
        )
        top = top.assign(Similarity = list(score(top['Word_Eval'],top['Synonym'])))
        trunc = top.sort_values(by='Count',ascending=False)

        print(f'Source:      {wp.source_name.capitalize()}')
        print(f'Description: {wp.description.capitalize()}')
        print(f'\n{top.head(10)}')

        # Find the max count for highlighting
        max_count = trunc['Count'].max()
        colors = ['red' if count == max_count else 'blue' for count in trunc['Count'].iloc[:20]]

        sns.set_style('whitegrid')
        sns.set_context("paper", font_scale=1.0)
        f, ax = plt.subplots(figsize=(6, 8), dpi=120)

        # Plot with custom color list
        g = sns.barplot(x='Count', y='Word', data=trunc.iloc[:20], palette=colors)

        import math
        if pd.isna(max_count):
            max_count_int = 0
        else:
            max_count_int = int(math.ceil(max_count))

        scale = list(range(0, max_count_int + 2, 2))
        plt.setp(ax, xticks=scale)
        g.set(title='Most Frequently Used Words\n', xlabel='Word Count', ylabel='Word List')
        plt.show()