import numpy as np
import json
import pickle
import random
import spacy


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizer_v2 import gradient_descent as SGD
from tensorflow.python.keras.models import load_model, save_model


#DEFINIMOS EL LENGUAGE CON EL CUAL VA A LEMMATIZAR SPACY
nlp = spacy.load("es_core_news_sm", disable = ['ner']) 


#ARCHIVO JSON QUE USAMOS COMO BASE DE ENTRENAMIENTO
intenciones= json.loads(open('XXXXXXXXX').read())


palabras = []
clases = []
documentos = []
ignorar_letras = ['!', '?', ',', '.']

for intencion in intenciones['intents']:
    clase= str(intencion['tag'])
    clases.append(clase)
    for w in intencion['patterns']:
        for i in range(0,len(ignorar_letras)):
             palabra1= str(w).replace(ignorar_letras[i],"",-1)
             w=palabra1
        palabra= str(w).split(sep=' ')    
        palabras.extend(palabra)       
        documentos.append((palabra,clase))
#print(f'{documentos}\n')
#print(f'{palabras}\n')
# print(f'{clases}\n')



#LLEVAMOS LAS PALABRAS A SU INFINITIVO CON EL LEMMATIZADOR DE SPACY


palabras = nlp(str(palabras).replace("'","",-1).replace("[","",-1).replace("]","",-1).replace(",","",-1))
palabras= [token.lemma_.lower() for token in palabras]
palabras = sorted(set(palabras))
clases = sorted(set(clases))


#GUARDAMOS LOS RESULTADOS EN ARCHIVOS
pickle.dump(palabras, open('palabras.pkl', 'wb'))
pickle.dump(clases, open('clases.pkl', 'wb'))
pickle.dump(documentos, open('documentos.pkl', 'wb'))



# DESDE ACA MODELAMOS LA DATA PARA QUE INGRESE A LA RED NEURONAL
training = []
output_empty = [0] * len(clases) #ceros como cantidad de clases tengamos


#CON ESTE LOOP CONVIERTO TODOS LOS DATOS EN 0 Y 1 PARA Q PUEDAN ENTRAR EN EL MODELO DE DL
for d in documentos:
    bag = []
    palabras_parametro = nlp(str(d[0]))
    palabras_parametro = [token.lemma_.lower() for token in palabras_parametro]
    for palabra in palabras:
        bag.append(1) if palabra in palabras_parametro else bag.append(0)
        output_row = list(output_empty)
        output_row[clases.index(d[1])] = 1
        training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)


train_x = list(training[:, 0]) #niveles de entrenamieto de la RN nivel 0 PALABRAS
train_y = list(training[:, 1]) #niveles de entrenamieto de la RN nivel 1 CLASES



# DESDE ACA EMPEZAMOS A CONTRUIR LA RED NEURONAL
model = Sequential() #modelo secuencial de DL
model.add(Dense(128, input_shape=(len(train_x[0]),   ), activation='relu')) #primera capa de densidad de 128 neuronas relu= f(x)= max(0, x)
model.add(Dropout(0.5)) #primera capa de salida dropout para no producir overfelling
model.add(Dense(64, activation='relu')) #segunda capa de densidad de 64 neuronas
model.add(Dropout(0.5)) # segunda capa de salida
model.add(Dense(len(train_y[0]), activation='softmax')) #ultima capa de salida de la red (funcion de activacion de cada neurona = softmax)

#Optimizador del modelo
sgd = SGD.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#definicion del loss function y optimizador
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('XXXXX.h5')
model.summary()
print("Listo! Largo2.0 está entrenado ;)") 
