import random
import json
import pickle
import numpy as np
import spacy
from marca_calibre import *

nlp = spacy.load("es_core_news_sm", disable = ['ner']) 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model

intenciones= json.loads(open('XXXXXXXXXX).read())
#marcacalibre= pd.read_excel("/Users/ramirofernandezdeullivarri/opt/rulo/newvenv/Chatbot/marca_calibre.xlsx")

palabras= pickle.load(open('palabras.pkl', 'rb'))
clases= pickle.load(open('clases.pkl', 'rb'))



#model = Sequential()
newmodel= load_model('XXXXXXXXX.h5')
newmodel.summary()

#frases ="Hola quiero una Stella Artois"

#LEMMATIZA LA FRASE (BUSCA LA REDUCCION MAS SIMPLE DE CADA PALABRA)
def frases_limpias(frases):
    palabrasdefrases= nlp(str(frases))
    palabrasdefrases= [token.lemma_.lower() for token in palabrasdefrases]
    return palabrasdefrases

#GENERA UNA LISTA DE 0 Y 1 DONDE SOLO ES 1 CUANDO LO Q SE PREGUNTO MACHEA CON LO Q ESTA A BD Y ENCONTRAMOS SU POSICION
def bolsa_de_palabras(frases):
    palabrasdefrases= frases_limpias(frases)
    bolsa= [0] *len(palabras)
    for p in palabrasdefrases:
     for i, palabra in enumerate(palabras):
          if palabra == p:
              bolsa[i]= 1
    return(np.array(bolsa))


#UTILIZA LA RED NEURAL YA ENTRENADA PARA "PREDECIR"/IDENTIFICAR LA "CLASES" DE ESA PALABRA ENCONTRADA
def tipo_prediccion(frases):
    bdp= bolsa_de_palabras(frases)
    resultado= newmodel.predict(np.array([bdp]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(resultado) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse= True)    
    return_list= []
    for r in results:
        return_list.append({'intent': clases[r[0]], 'probabilidad': str(r[1])})
    return return_list   

#APARTIR DE LA CLASE ENCONTRADA, DEVUELVE UNA RESPUESTA ALEATORIA 
def get_response(return_list,intenciones):
    try:
        tag = return_list[0]['intent']
        intenciones_lista= intenciones['intents']
        for i in intenciones_lista:
            if i['tag']==tag:
                result= random.choice(i['responses'])
                break
    except IndexError:
        result="No entiendo lo que me estas diciendo"
    
    return result


#AL IDEI IDENTIFICAR ALGUNA CLASE EN PARTICULAR DISPARA OTRO CODIGO
mapping= {'Marcas': marca_calibre,
          'Negocio': marca_calibre,
          'Calibres': marca_calibre
          }

#GENERA UNA BUSQUEDA O LLAMADA DE MARCA CALIBRE
def request(frases):
    return_list = tipo_prediccion(frases)
    if return_list[0]['intent'] in mapping.keys():
        return mapping[return_list[0]['intent']](frases)
    else:
        return get_response(return_list,intenciones)



print("Charbot y cargado listo para usar!")   

#LOOP QUE INICIA EL CHATBOT
while True:
    frases= input("YO: ")
    print(request(frases))


print(bolsa_de_palabras(frases))







