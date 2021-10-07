import unicodedata
import string
import re
import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import spacy
import spacy_streamlit
from spacy_streamlit import visualize_tokens

# this is is the function to form a dense matrix - after countvectorizer
def dense(x):
    y = x.todense()
    return y

# loading the trained models
pickle_in_fed = open('model_federal_histgb_bow_pipeline.pkl', 'rb') 
classifier_fed = pickle.load(pickle_in_fed)

pickle_in_est = open('model_estadual_histgb_tfidf_pipeline.pkl', 'rb') 
classifier_est = pickle.load(pickle_in_est)

@st.cache(suppress_st_warning=True)

# defining the function which will make the prediction using the data which the user inputs 
def prediction(text, option):   
 
    # Pre-processing user input    
    
    #STOPWORDS
    mystopwords = ['de', 'a', 'o', 'que', 'e', 'é', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos', 'teriam', 'jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', 'janeiro', 'fevereiro', 'marco', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro', 'art', 'artigo', 'apos', 'após', 'sobre', 'inc.', 'inciso', 'nao', 'alinea', 'paragrafo', 'caput', 'conforme', 'disposto', 'deste', 'ate', 'nas', 'naquela', 'naquelas', 'naquele', 'naqueles', 'naquilo', 'nessa', 'nessas', 'nesta', 'nestas', 'nesse', 'nesses', 'neste', 'nestes', 'nisso', 'nisto', 'daquela', 'daquelas', 'daquele', 'daqueles', 'daquilo', 'dispostos', 'disso', 'nuns', 'numas', 'uns', 'umas', 'desse', 'desses', 'destes', 'inc', 'ja', 'ha', 'hao', 'nao', 'nos', 'sao', 'so', 'tambem', 'tera', 'terao', 'teriamos', 'tiveramos', 'tivessemos', 'tem', 'tinhamos', 'uns', 'umas', 'voce', 'voces', 'eramos', 'dentre', 'ser', 'sera', 'serao', 'seriamos', 'esta', 'vossa', 'vossas', 'vosso', 'vossos', '&']
    
    #TRANSLITERATION: removing accentuation, male ordinal signal (º) and degree signal(°).
    processing = unicodedata.normalize("NFD", text)
    processing = processing.encode("ascii", "ignore")
    processing = processing.decode("utf-8")
    
    #CLEANING LEGAL TEXTS
    processing = ''.join([i for i in processing if not i.isdigit()]) #Cleaning digits
    processing = re.sub(r"http\S+", "", processing).lower().replace('.',' ').replace(',',' ').replace(';',' ').replace('-',' ').replace(':',' ').replace(')',' ').replace('(',' ').replace('/',' ').replace('\\', ' ').replace('""', ' ').replace('§',' ').replace('º',' ').replace('°',' ').replace('nº',' ').replace('n°',' ').replace('%',' ').replace('*',' ').replace('?',' ').replace('"',' ').replace('_',' ').replace('<', ' ').replace('>', ' ').replace('[', ' ').replace(']', ' ').replace('|', ' ').replace('+', ' ').replace('=', ' ') #Cleaning data
    processing = re.sub(r'(?=\b[mcdxlvi]{1,6}\b)m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})', ' ', processing) #Cleaning roman numerals (ex.: 'incisos')
    processing = re.sub(r'\s[a-z]\s|^[a-z]\s|\s[a-z]$', ' ', processing) #Cleaning isolate letters a-z (ex.: 'alíneas')
    processing = re.sub(r'([bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz]*)', ' ', processing) #Cleaning consoant sequences (ex.: hash code)
    processing = ' '.join([i for i in processing.split() if not i in mystopwords])
        
    #LEMMATIZATION
    nlp = spacy.load('pt_core_news_sm')
    doc = nlp(processing)
    lemma = ' '.join([word.lemma_.lower() for word in doc])
    #visualize_tokens(doc, attrs=["text", "pos_", "dep_", "ent_type_", "lemma_"])
                
    #Making predictions 
    if option == 'Federal':
        prediction = classifier_fed.predict([lemma])
        prediction_proba = classifier_fed.predict_proba([lemma])
    else:
        prediction = classifier_est.predict([lemma])
        prediction_proba = classifier_est.predict_proba([lemma])
     
        
    if prediction == 1:
        pred = 'RELEVANTE'
        pred_proba = round(prediction_proba[0][0]*100, 0)
    else:
        pred = 'IRRELEVANTE'
        pred_proba = round(prediction_proba[0][1]*100, 0)
        
        # se proba > 80% retorna pred
        # else retorna indefinido
    return pred, pred_proba
    
    
    '''if prediction == 1:
        pred = 'RELEVANTE'
    else:
        pred = 'IRRELEVANTE'
        
        # se proba > 80% retorna pred
        # else retorna indefinido
    return pred, prediction_proba[0][0]*100'''

  
 # this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:green;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Classificador de Atos Legais</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
     
    text = st.text_area("Cole o texto do ato legal")
    result =""
      
    # to select the legal text abrangency
    option = st.selectbox(
    'Indique a abrangência do ato legal', ['', 'Estadual', 'Federal'])
    
    
    # when 'Predict' is clicked, make the prediction and store it 
    if option == 'Estadual' or option == 'Federal':
        if st.button("Classificar"): 
            result, proba = prediction(text, option)
            st.success('Para a Cia, este ato legal é {}, com {}% de probabilidade.'.format(result, proba))

        
if __name__=='__main__':
    main()
    