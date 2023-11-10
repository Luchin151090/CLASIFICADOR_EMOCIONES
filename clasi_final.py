import pandas as pd
import spacy  # Spacy nos ayuda a tokenizar y  es mejor que las librerias de token normal
import re
from transformers import BertTokenizer, BertForSequenceClassification,AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch

#-------------CONFIGURAR DISPOSITIVO PC
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------DIVISIÓN DEL DATASET EN UN CONJUNTO ESPECÍFICO
# Cargo el archivo
data = pd.read_csv(r'D:\\clasificador_emociones\\train.csv')
print("---------columnas---------")
print(data.columns)

# Selecciono solo los registros q necesito
newdata_set = data.head(80000)

# Guardar en una nueva carpeta
newdata_set.to_csv("movie_data.csv",index=False)


#--------------LIMPIEZA DE DATOS CON EXP. REGULARES
# Cargar los comentarios
data = pd.read_csv("movie_data.csv")

# Limpiamos
def clean_review(comentario):
    if isinstance(comentario,str):
        comentario_limpio = re.sub(r'[^A-Za-z ]+', '', comentario)
        comentario_limpio = comentario_limpio.lower()
        return comentario_limpio
    else:
        return ''
reviews = data['review_text'].apply(clean_review)

#---------------UTILIZAMOS SPACY PARA LA TOKENIZATION y ELIMINAMOS LOS STOPWORDS "el, la, etc"
#-------------SPACY USA UNA TOKENIZATION MUY GENERAL , BASADO EN DICCIONARIOS


# Modelo spacy para idioma
#nlp = spacy.load("en_core_web_sm")

# Lista vacia para tokens
#tokenized_reviews = []

# Eliminando los stopwords
#def remove_stopwords(tokens):
#    return [word for word in tokens if word not in nlp.Defaults.stop_words]

#for review in reviews:
#    doc = nlp(review)
#    tokens = []
#    for token in doc:
#        tokens.append(token.text)
#    tokenized_reviews.append(tokens)

#comentarios_tokenizados = tokenized_reviews
#print(comentarios_tokenizados)

#-----------------UTILIZAMOS BERT PARA TOKENIZAR, YA QUE HACE UNA MEJOR TOKENIZATION, POR ENTENDER MEJOR NLP

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 128  # Puedes ajustar esto según tus necesidades

tokenized_reviews = []

for review in reviews:
    tokens = tokenizer(review, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    tokenized_reviews.append(tokens)


# Codificar las etiquetas
labels = data['class_index'].tolist()

# Aquí usamos la división 80% para entrenamiento y 20% para pruebas
X_train_texts, X_test_texts, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Tokenizar las oraciones
X_train_encodings = tokenizer([review for review in X_train_texts], truncation=True, padding=True, max_length=max_length, return_tensors='pt', return_attention_mask=True)
X_test_encodings = tokenizer([review for review in X_test_texts], truncation=True, padding=True, max_length=max_length, return_tensors='pt', return_attention_mask=True)

# Convertir a tensores de PyTorch
X_train = X_train_encodings['input_ids']
X_test = X_test_encodings['input_ids']

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)




# etiquetas de clases
num_classes = 5

# Cargar el modelo pre-entrenado
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)  # 5 clases emocionales

# Tenemos que usar CUDA
model.to('cuda')

# Coloca los datos en un formato Dataset de PyTorch
train_data = TensorDataset(X_train, y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

# Parámetros de entrenamiento
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

num_epochs = 7

# Entrenamiento del modelo
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        inputs, labels = batch
        optimizer.zero_grad()

        # inputs a GPU
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Imprimir la pérdida promedio por época
    print(f'Época {epoch+1}, Pérdida: {total_loss/len(train_dataloader)}')
    

#---------------FINALMENTE GUARDAMOS EL MODELO ENTRENADO EN UNA RUTA
# Esto para que se pueda usar con otros comentarios
model.save_pretrained(r"D:\\clasificador_emociones")
