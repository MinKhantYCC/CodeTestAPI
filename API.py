import numpy as np
# import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
# from tensorflow.keras.preprocessing.text import Tokenizer
from mm_tokenize import Tokenize
import streamlit as slt

sentences = []
with open('sample.txt', 'r', encoding='utf-8') as f:
    for text in f:
        txt1 = text.split('။')
        for t in txt1:
            if t!='' and t!='\n':
                sentences.append(t.strip())

# print(sentences)

mmtk = Tokenize()
vectorizer = TfidfVectorizer(tokenizer=lambda x: mmtk.tokenize_word(x,lang='mm',form='syllable'))

def compute_similarity(a, b):
    tfidf = vectorizer.fit_transform([a, b])
    return ((tfidf*tfidf.T).toarray())[0,1]

def answer_question(question, verbose=1):
    best_sentences = ''
    output_sentences = []
    similarities = []
    for sentence in sentences:
        sim = compute_similarity(question, sentence)
        if sim >= 0.05:
            output_sentences.append(sentence)
            similarities.append(sim)
    if len(output_sentences)>4:
        for ind, sim in enumerate(similarities):
            if sim >= 0.15:
                best_sentences += output_sentences[ind]+'။ '
    else:
        for sents in output_sentences:
            best_sentences += sents + '။ '
            
    return best_sentences

slt.title('You Ask, I Answer :sunglasses:')
sentence=''
for sent in sentences:
    sentence += sent
button_clicked = False
question = slt.text_input('Ask a question', value='', placeholder='Question',key = 'qus')
if question != '':
    best_sentences = answer_question(question)
button_clicked = slt.button('Generate',key = 'ans')
if button_clicked:
    slt.write('Answer: ' + best_sentences)
else:
    slt.write('')
if slt.button('Source', key= 'src'):
    slt.write(sentence)
if slt.button('Clear', key= 'clr'):
    sentence = ''
    best_sentences = ''
