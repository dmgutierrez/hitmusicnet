from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from langdetect import detect
import pycountry
import pyphen
import numpy as np
import pandas as pd

class TextFeatures():
    def __init__(self, corpus):
        self.corpus = corpus
        self.language, self.code = self.detect_language()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.d_syl = None
        self.stopW = None
        self.sentences = []
        self.total_sentences = -1
        self.words=[]
        self.total_words = -1
        self.mean_words_sentence = -1
        self.mean_syllables_word = -1
        self.vocabulary = {}
        self.vocabulary_wealth = -1
        self.sentence_similarity = -1
        self.df_text_features = None
        
    def get_sentences(self):
        sentences = []
        total_sentences = 0
        try:
            if self.corpus !='-99' and self.corpus !='':
                sentences = self.corpus.split('\n')
                sentences = [s for s in sentences if s !='']
                total_sentences = len(sentences)
        except Exception as e:
            print(e)
        return sentences, total_sentences
    
    def get_words(self):
        words = []
        total_words = 0
        try:
            if self.corpus !='-99' and self.corpus !='':
                tokenized_words = self.tokenizer.tokenize(self.corpus)
                total_words = len(tokenized_words)
                # Remove stopwords
                words = [word for word in tokenized_words if word not in self.stopW]

        except Exception as e:
            print(e)
        return words, total_words
    
    def get_vocabulary_dictionary(self):
        vocabulary = {}
        try:
            if self.words:
                wordfreq = [self.words.count(w) for w in self.words]
                vocabulary = dict(zip(self.words,wordfreq))
        except Exception as e:
            print(e)
        return vocabulary
    
    def get_vocabulary_wealth(self, max_cum=0.80):
        try:
            min_words = 0 
            vocabulary_wealth = 0
            if max_cum<=1 and self.words and self.vocabulary:
                word_freq = np.array(sorted(list(self.vocabulary.values()),
                                            reverse=True))
                # Cumulative sum of the words
                cumWords = word_freq.cumsum()
                maximum_val = int(cumWords[-1]*max_cum)
                for index, p in enumerate(cumWords):
                    if p >=maximum_val:
                        min_words = index
                        break
                vocabulary_wealth = round(min_words/len(word_freq),2)
        except Exception as e:
            print(e)
        return vocabulary_wealth
    
    def detect_language(self):
        lang_name=None
        code = None
        try:
            if self.corpus !='-99' and self.corpus != '': 
                code = detect(self.corpus)
                # Get language name
                country = pycountry.languages.get(alpha_2=code)
                lang_name = country.name.lower()
        except Exception as e:
            print(e)
        return lang_name, code
    
    def get_mean_words_per_sentence(self):
        mean_sent = 0
        words = []
        try:
            if self.sentences:
                for s in self.sentences:
                    tokenized_words = self.tokenizer.tokenize(s)
                    # Remove stopwords
                    words.append(len([word for word in tokenized_words if word not in self.stopW]))
                mean_sent = round(np.mean(words),2)
        except Exception as e:
            print(e)
        return mean_sent
    
    def get_mean_syllables_per_word(self):
        mean_syl = 0
        try:
            if self.words and self.d_syl is not None:
                mean_syl = round(np.mean([len(self.d_syl.inserted(w).split('-')) for w in self.words]),2)
        except Exception as e:
            print(e)
        return mean_syl
    
    def compute_sentence_similarity(self, similarity_thres = 0.40):
        sentence_sim = 0
        try:
            if self.sentences:
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(self.sentences)
                cos_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
                if len(self.sentences)>1:
                    tri_upper_diag = np.triu(cos_matrix, k=1)
                    val_mat = tri_upper_diag[np.where(tri_upper_diag>=similarity_thres)]
                    
                    diag_val = cos_matrix.shape[0]
                    total_elements = (cos_matrix.shape[0]*cos_matrix.shape[1]-diag_val)/2
                    sentence_sim = (len(val_mat)/total_elements)
        except Exception as e:
            print(e)
        return sentence_sim
    
    
    def run(self, index, max_cum=.80, mu=.40):
        ok = False
        try:
            if self.code is not None and self.language is not None:
                self.d_syl = pyphen.Pyphen(lang=self.code)
                self.stopW = set(stopwords.words(self.language))
                self.sentences, self.total_sentences = self.get_sentences()
                self.words, self.total_words = self.get_words() 
                # Mean Words per Sentence
                self.mean_words_sentence = self.get_mean_words_per_sentence()
                
                # Mean Syllables per Word
                self.mean_syllables_word = self.get_mean_syllables_per_word()
                # Mean Vocabulary Wealth
                self.vocabulary = self.get_vocabulary_dictionary()
                self.vocabulary_wealth = self.get_vocabulary_wealth(max_cum=max_cum)
                # Sentence Similarity
                self.sentence_similarity = self.compute_sentence_similarity(similarity_thres=mu)
                
            # Data Dictionary
            data = {'n_sentences':self.total_sentences,
                    'mean_words_sentence': self.mean_words_sentence,
                    'n_words':self.total_words,
                    'mean_syllables_word':self.mean_syllables_word,
                    'sentence_similarity':self.sentence_similarity,
                    'vocabulary_wealth':self.vocabulary_wealth}
            
            self.df_text_features = pd.DataFrame(data,index=[index])
            ok=True
        except Exception as e:
            print(e)
        return ok
