# -*- coding: utf-8 -*-
"""
Created on Sun Jan 3 23:05:18 2021

@author: Afaf et Bochra
"""
import re
import praw
import urllib.request
import xmltodict  
from Classes import Document, Corpus, Author, RedditDocument, ArxivDocument
import datetime as dt 
import pandas as pd
from PIL import Image, ImageTk #conda install -c anaconda pillow 
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))    
####### interface ########
import pandas as pd
from tkinter import *
from tkinter import ttk
#from tkinter.messagebox import *
import chardet


################################## Création du Corpus ##################################

corpus = Corpus("Corona")

reddit = praw.Reddit(client_id='fJ2X6dGvzlqhxA', client_secret='GKXiZ-vxjKkFrXM-x5X8rZ_H6Nk', user_agent='td1App')
hot_posts = reddit.subreddit('Coronavirus').hot(limit=100)
for post in hot_posts:
    datet = dt.datetime.fromtimestamp(post.created)
    txt = post.title + ". "+ post.selftext
    doc = RedditDocument(datet,
                   post.title,
                   post.author_fullname,
                   txt,
                   post.url,0)
    corpus.add_doc(doc)



url = 'http://export.arxiv.org/api/query?search_query=all:covid&start=0&max_results=100'
data =  urllib.request.urlopen(url).read().decode()
docs = xmltodict.parse(data)['feed']['entry']

for i in docs:
    datet = dt.datetime.strptime(i['published'], '%Y-%m-%dT%H:%M:%SZ')
    try:
        author = [aut['name'] for aut in i['author']][0]
    except:
        author = i['author']['name']
    txt = i['title']+ ". " + i['summary']
    doc = ArxivDocument(datet,
                   i['title'],
                   author,
                   txt,
                   i['id'],0
                   )
    corpus.add_doc(doc)
    

######################### Projet ############################

#Initialisation des 3 texteS pour reccuperer les informations souhaité
# Documents/Articles toute classe conondu
txt = ""
# Articles issus de Reddit
reddit_txt = ""
# articles issus d’Arxiv
arxiv_txt = ""


#On récupère les informations souhaités selon le type de document 
for  i in range (corpus.ndoc) :    
    doc = corpus.get_doc(i)
    txt += doc.get_text()
    if doc.getType() == "reddit" : 
        reddit_txt += doc.get_text()
    else : 
            arxiv_txt += doc.get_text()


# Stop word  : https://pythonspot.com/nltk-stop-words/
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')



#On s'attaque maintenant au traitement de nos données
def traitement(string):
    #On rend le contenu digeste et eviter les doublons et les données qui ne sont pas cohérente
    string = string.lower()
    string = string.replace('\n', ' ')
    string = string.replace('\r', ' ')
    string = string.translate ({ord(c): " " for c in "!@'#$,%^&*()[]{};:,.-/<>?\|`~=_+1234567890"})
    string = re.sub(" +", " ",string) ### Eviter les doubles espaces 
    
    #grâce à la librairie on va prendre les stopword et filtrer nos donnée en conséquence
    stopWords = set(stopwords.words('english')+stopwords.words('french'))
    words = word_tokenize(string)
    wordsFiltered = []
    
    for w in words:
        if w not in stopWords:
             if w != "u": # u n'est pas utile on le retrouve dans le top
                wordsFiltered.append(w) # on prend que les mots intéressant
    
    #print(wordsFiltered)
    liste_mots = set(wordsFiltered)
    
    df_mots = pd.DataFrame(columns=['Mots','Occurence'])
    i = 0
    #On va créer un data frame ou on reprend tous ces mot ainsi que leur occurences dans les textes
    for mot in liste_mots : 
        df_mots.loc[i,'Mots'] = mot   
        wordsFiltered.count(mot)
        df_mots.loc[i,'Occurence'] = wordsFiltered.count(mot)
        i += 1
        #df_mots.loc[i,'Periode'] =
    #on tri le tableau pour avoir le top au debut : les mots les plus représentés   
    df_mots = df_mots.sort_values(by = 'Occurence',ascending = False)
    return df_mots

#On applique la fonction au 3 textes
df_txt = traitement(txt)
df_reddit = traitement(reddit_txt)
#l'affichage des corpus 
print("df_reddit")
print(df_reddit)
df_arxiv = traitement(arxiv_txt)
print("df_arxiv")
print(df_arxiv)

#recuperer l'occurence du mot :combien de fois le mot est utiliser
#dans le corpus 
numrepcorpus1= {df_reddit["Mots"][i]:df_reddit["Occurence"][i] for i in range(len(df_reddit))}
numrepcorpus2= {df_arxiv["Mots"][i]:df_arxiv["Occurence"][i] for i in range(len(df_arxiv))}


########### ------------------
#conda install -c conda-forge wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

######## NUAGE DE MOTS Pour réalisé l'importances des mots dans les textes
wordCloud = WordCloud()
wordCloud.generate(txt) 
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("txt.jpg")
#nuage de mot reddit
wordCloud_reddit = WordCloud()
wordCloud_reddit.generate(reddit_txt) 
plt.imshow(wordCloud_reddit, interpolation='bilinear')
plt.axis("off")
plt.savefig("txt_reddit.jpg")
#nuage de mot arxiv
wordCloud_arxiv = WordCloud()
wordCloud_arxiv.generate(arxiv_txt) 
plt.imshow(wordCloud_arxiv, interpolation='bilinear')
plt.axis("off")
plt.savefig("txt_arxiv.jpg")


################## COMPARAISON #######################

#-----------------------> TROUVER UNE METHODE QUI COMPARE LES DEUX DATA FRAME

#fonction pour calculer la frequence TF d'un mot dans le corpus (https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76)
def freqTF(wordDict, decoup):
    tfDict = {}
    decoupCount = len(decoup)
    for word, count in wordDict.items():
        tfDict[word] = count / float(decoupCount)
    return tfDict

#decouper les corpus en mot 
decoup1 = reddit_txt.split(" ")
decoup2 = arxiv_txt.split(" ")
#on applique la fonction freqTF sur les mots
tfcorpus1=freqTF(numrepcorpus1,decoup1)
tfcorpus2=freqTF(numrepcorpus2,decoup2)

#on prepare le tableau pour l'affichage 
df_reddit["freq_tf"]=[" "]*len(df_reddit)
df_arxiv["freq_tf"]=[" "]*len(df_arxiv)


for i in range(len(df_reddit)):
    df_reddit["freq_tf"][i] = tfcorpus1[  df_reddit["Mots"][i]   ]
    
for i in range(len(df_arxiv)):
    df_arxiv["freq_tf"][i] = tfcorpus2[  df_arxiv["Mots"][i]   ]
   


#fonction pour calculer la frequence IDF d'un mot 
def freqIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        
        for word, val in document.items():
            if val > 0:
                    idfDict[word] += 1
            
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


#creer la liste des mots en commun 
liste = set(decoup1).union(set(decoup2))
#fonction pour affecter un vecteur a chaque corpus --> la repetition des mots dans le corpus 
numrepcorpus1 = dict.fromkeys(liste, 0)
for word in decoup1:
    numrepcorpus1[word] += 1
    
numrepcorpus2 = dict.fromkeys(liste, 0)
for word in decoup2:
    numrepcorpus2[word] += 1


#on applique la fonction IDF sur les mots
idf=freqIDF([numrepcorpus1,numrepcorpus2])
idf

#preparation de l'affichage du tableau 
df_reddit["freq_idf"]=[" "]*len(df_reddit)
df_arxiv["freq_idf"]=[" "]*len(df_arxiv)


for i in range(len(df_reddit)):
    try :
        df_reddit["freq_idf"][i] = idf[  df_reddit["Mots"][i]   ]
    except:
        None
for i in range(len(df_arxiv)):
    try :
        df_arxiv["freq_idf"][i] = idf[  df_arxiv["Mots"][i]   ]
    except:
        None    
   

#fonction pour calculer la frequence TFIDF d'un mot
def freqTFIDF(tfdecoup, idf):
    tfidf = {}
    for word, val in tfdecoup.items():
        try :
            tfidf[word] = val * idf[word]
        except :
            None
    return tfidf

#on calcule la frequence sur les mots
tfidf1 = freqTFIDF(tfcorpus1, idf)
tfidf2 = freqTFIDF(tfcorpus2, idf)
df = pd.DataFrame([tfidf1, tfidf2])



#preparation de l'affichage du tableau
df_reddit["freq_tfidf"]=[" "]*len(df_reddit)
df_arxiv["freq_tfidf"]=[" "]*len(df_arxiv)


for i in range(len(df_reddit)):
    try :
        df_reddit["freq_tfidf"][i] = tfidf1[  df_reddit["Mots"][i]   ]
    except:
        None
for i in range(len(df_arxiv)):
    try :
        df_arxiv["freq_tfidf"][i] = tfidf2[  df_arxiv["Mots"][i]   ]
    except:
        None    
   
#l'affichage des deux tableau de nos corpus 
# on va afficher les mots de chaque corpus , l'occurence du mot 
#avec les frequences , TF ,IDF et TFIDF   
print("####################  REDDIT CORPUS   ###################")
print(df_reddit)
print("####################  ARXIV CORPUS   ###################")
print(df_arxiv)




#Boucle pour trouver les mots commun 
mots_commun=[]
a=[]
for i in range ( len(df_reddit) ) :
    
    if df_reddit["Mots"][i] in list( df_arxiv["Mots"]  )    :
        mots_commun.append(df_reddit["Mots"][i])
        a.append(df_reddit["Mots"][i])
    
    if (len(mots_commun) > 10 ) :
        break
        
 #affichage des mots commun        
print("####################   Les mots communs   ###################")

print(mots_commun)
print(a)

#afficher les freqs dans un tableau  
df.head()



################################# fonction pour l'affichage d'information d'un mot choisi ##################################

def info_mot():
    #on recupere le mot rentrer par l'utilisateur 
    mot= entreemot.get()
#l'inisialisation de mes variables
    #print(mot)
    occ_corpus1=0
    occ_corpus2=0
    tfidf1=0
    tfidf2=0
    #boucle pour recuperer l'occurence du mot dans le 1er corpus
    for i in range(len(df_reddit))  :
        if   df_reddit["Mots"][i] == mot :
            occ_corpus1 = df_reddit["Occurence"][i]
            tfidf1 = df_reddit["freq_tfidf"][i]
    #boucle pour recuperer l'occurence du mot dans le 2eme corpus       
    for i in range(len(df_arxiv))  :
        if   df_arxiv["Mots"][i] == mot :
            occ_corpus2 = df_arxiv["Occurence"][i]
            tfidf2 = df_arxiv["freq_tfidf"][i]
   #la forme sous la quelle on va afficher les resultat
    occ_corpus1 = "occurence dans le corpus1:" + str(occ_corpus1) + "\n"
    occ_corpus2 = "occurence dans le corpus2:" + str(occ_corpus2) + "\n"
    tfidf1 = "tfid dans le 1er corpus:" + str(tfidf1) + "\n"
    tfidf2 = "tfid dans le 2eme corpus:" + str(tfidf2) + "\n"
    return( occ_corpus1 , occ_corpus2 , tfidf1  , tfidf2)

#x=info_mot(mot)


######################################### TEMPORALITE  ####################################################################
#Nous avons décider de travailler sur le top des mots de chaque texte
df1=df_reddit["Mots"].head(6)
df2=df_arxiv["Mots"].head(6)
frames = [df1, df2]
#création du dataFrame
premier_mots = pd.concat(frames).drop_duplicates()
print(premier_mots)

df_periode = pd.DataFrame(columns=["Mots","Periode"])    

      #On prend ces mots et on cherche la date time d'apparition'
x=0
for i in range(corpus.ndoc):
    doc2 = corpus.get_doc(i)
    txt = ""
    txt = doc2.get_text()
    for j in premier_mots : 
        if j in txt:
            df_periode.loc[x,'Mots'] = j
            # On va stocker que les mois et les années pour une meilleure lisibilité
            df_periode.loc[x,'Periode']=str(doc2.get_date().month) + '-' + str(doc2.get_date().year)
            x+=1

print(df_periode)
#On crée le plot : scatter pour bien visualiser les périond où les mot on été utilisé
fig = df_periode.plot('Mots','Periode',kind='scatter',colormap='viridis',  figsize=(20, 16), fontsize=26).get_figure()
fig.savefig('temp.jpg')



################################## interface ##################################

#================ Fenêtre et style ===============
root = Tk()
root.title("Projet programmation Python")
style = ttk.Style()
style.theme_use('vista')
root.configure(background='#49A')


########### Fonctions bouttons
def valider():
  sortie1.delete("1.0", "end")
  sortie1.insert(INSERT, mots_commun)
  
def valider1():
  sortie1.delete("1.0", "end")  
  sortie1.insert(INSERT, info_mot())  
  
def valider2():
  sortie1.delete("1.0", "end")  
  sortie1.insert(INSERT, df_reddit,"   ESPACE   ",df_arxiv)  
  

pos = "0.0"
sortie1 = Text(root, width=70,height=10)
sortie1.place(x=700,y=420)

### Affichage du graphique de la temporalité
def tempo():
    fen1=Toplevel()
    fen1.title("Temporalité des mots les plus présent dans nos textes")
    im1 = PhotoImage(file="/temp.jpg")
    B7=Label(fen1, image=im1)
    B7.grid(row=0,column=1)
    return


#### Affichafe de nos word cloud
def word_cloud():
    fen1=Toplevel()
    fen1.title("Word Cloud Texte entier")
    im1 = PhotoImage(file=os.chdir(os.path.dirname(os.path.abspath(__file__))+"\\txt.jpg"))
    B4=Label(fen1, image=im1)
    B4.grid(row=0,column=1)
    fen2=Toplevel()
    fen2.title("Word Cloud Reddit")
    im2 = PhotoImage(file='/txt_reddit.jpg')
    B5=Label(fen2, image=im2)
    B5.grid(row=0,column=1)
    fen3=Toplevel()
    fen3.title("Word Cloud Arxiv")
    im3 = PhotoImage(file='\txt_arxiv.jpg')
    B6=Label(fen3, image=im3)
    B6.grid(row=0,column=1)
    


#================== CONFIGURATION DES FRAMES --> root =====================

F1=ttk.Frame(root, borderwidth=2)
F1.grid(row=0, column=0,columnspan=3)
F1.config(width=200,height=200)

F2=ttk.Frame(root, borderwidth=2)
F2.grid(row=1, column=0,columnspan=3)

F3=ttk.Frame(root, borderwidth=2)
F3.grid(row=2, column=0,columnspan=3)

F4=ttk.Frame(root, borderwidth=2)
F4.grid(row=3, column=0,columnspan=3)


F31=ttk.Frame(F3, borderwidth=2)
F31.grid(row=0, column=1,columnspan=3)


################################# label et entrée F1 ##################################

corpus1 = Label(F1,text="REDDIT TEXTE  :",fg="black",bg='#49A')
corpus1.grid(row=0,column=0, sticky='snew')
entree1 = Text(F1,  width = 40)
entree1.grid(row=0, column=1,sticky='snew', padx=20, pady=10)
entree1.insert(INSERT, reddit_txt)

corpus2 = Label(F1,text="ARXIV TEXTE  :",fg="black",bg='#49A')
corpus2.grid(row=0,column=2, sticky='snew')
entree2 = Text(F1,  width = 40)
entree2.grid(row=0, column=3,sticky='snew', padx=20, pady=10)
entree2.insert(INSERT, arxiv_txt)


#================== F2 ======================


txt = Label(F2, text = "Tapez le mot à analyser:",fg="black",bg='#49A')
txt.grid(row=1,column=0, sticky='snew')
entree3 = Text(F2,  width = 20, height = 10)
entree3.grid(row=1, column=1,columnspan=3,sticky='snew', padx=20, pady=10)





#================== configuration des bouton : F3 ======================

################################# bouton mots commun ##################################
motcommun = ttk.Button(F3, text="Mots commun")
motcommun.grid(row=1, column=0, sticky='snew', ipadx=5, ipady=5, padx=15, pady=15)
motcommun.config(command=valider,width=20)



################################# bouton analyser ##################################
comparer = ttk.Button(F3, text="Analyser")
comparer.grid(row=1, column=1, sticky='snew', ipadx=5, ipady=5, padx=15, pady=15)
comparer.config(command=valider2,width=20)



################################# bouton valider ##################################
valider= ttk.Button(F3, text="Valider")
valider.grid(row=1, column=2, sticky='snew', ipadx=5, ipady=5, padx=20, pady=15)
valider.config(command=valider1,width=15)


#################### bouton afficher les Temporalité #########################
temp= ttk.Button(F3, text="Temporalité des mots les plus présent dans nos textes")
temp.grid(row=1, column=3, sticky='snew', ipadx=5, ipady=5, padx=15, pady=15)
temp.config(command=tempo,width=45)



#################### bouton afficher les Words Clouds #########################
butWC = ttk.Button(F3, text="Afficher les Words Clouds")
butWC.grid(row=1, column=4, sticky='snew', ipadx=5, ipady=5, padx=15, pady=15)
butWC.config(command=word_cloud,width=25)



#==================Button Quitter : F4 ======================
Bu7 = ttk.Button(F4,text='Quitter')
Bu7.grid(row=0,column=0,sticky=E)
Bu7.config(command=lambda: root.destroy(),width=15)




root.mainloop()

root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.geometry("1000x700")
root.mainloop()





