#%%
from nltk.tag import StanfordPOSTagger
import os

#Cargamos el texto a analizar

texto = open("data/noticia.txt",'r').read()

#Se carga el tagger entrenado desde Stanford

Tagger = StanfordPOSTagger("data/stanford-postagger/models/spanish-ud.tagger","data/stanford-postagger/stanford-postagger.jar")

print(texto.split("\n"))

tagged_text = Tagger.tag(texto.split())

#Se imprimen las palabras

print(tagged_text)
# %%
