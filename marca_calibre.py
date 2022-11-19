import numpy as np
import json
import pickle
import spacy
import pandas as pd



#frases = "tenes el codigo de corona de litro ?"

def marca_calibre(frases):
    #frases= input("Marca calibre de que producto?: ")
    df = pd.read_excel("/Users/ramirofernandezdeullivarri/opt/rulo/newvenv/Chatbot/marca_calibre.xlsx")
    nlp = spacy.load("es_core_news_sm", disable=['ner'])
    documentos = pickle.load(open('documentos.pkl', 'rb'))
    clasex = []

    #APARTIR DE LA FRASE IDENTIFICA QUE CLASE ES 
    doc = nlp(frases)
    doc = [token.lemma_.lower() for token in doc if len(token) >= 3]
    for t in doc:
        bag = []
        output_row = []
        for d in documentos:
            palabras_parametro = nlp(str(d[0]))
            palabras_parametro = [token.lemma_.lower() for token in palabras_parametro]
            if t in palabras_parametro:
                bag.append(str(t).replace("[", "", -1).replace("]", "", -1))
                output_row.append(str(d[1]).replace("[", "", -1).replace("]", "", -1))
                clasex.append((bag, output_row))
            else:
                pass
    #print(clasex)
    #print(len(clasex))

    #SON LOS NOMBRES DE LAS COLUMNAS DE LA BD DE EXCEL
    calibre = []
    marca = []
    negocio = []
    estilo = []
    clase = {"Negocio": negocio,
             "Marcas": marca,
             "Calibres": calibre,
             "Estilo": estilo
             }
    for k, v in clase.items():
        for res in clasex:
            if res[1] == [k]:
                v.append(str(res[0]).replace("[", "", -1).replace("]", "", -1).replace("'", "", -1))
            else:
                pass
    #print(f"{clase}\n")

    #BUSCA EL DATAFRAME APARTIR DE LOS FILTROS/COLUMNAS ENCONTRADAS
    for k, v in clase.items():
        if len(v) > 0:
            df = df.loc[df[str(k)].isin(v)]
        else:
            pass

    #DEVUELVA LA RESPUESTA CORRECTA    
    muestra=['Codigo']
    [muestra.append(k) for k,v in clase.items() if len(v)>0]    
    df= df[muestra]
    return f"Te paso la info que me pediste! \n {df}"

#print(marca_calibre())
