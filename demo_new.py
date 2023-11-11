import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Configurar el dispositivo (cuda o cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo pre-entrenado
model = BertForSequenceClassification.from_pretrained("D:\\clasificador_emociones")
model.to(device)  # Mover el modelo a la GPU si está disponible

# Inicializar el tokenizador
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Cargar el nuevo dataset
new_data = pd.read_csv("D:\\clasificador_emociones\\steam_new_data.csv")

# AQUI ESCOGEMOS LA CANTIDAD DE ARCHIVOS QUE QUEREMOS 
parte_data = new_data.head(1000)


# Crear una lista para almacenar las predicciones
predicted_labels = []

# Iterar sobre los comentarios en el nuevo dataset
for comment in parte_data['review']:
    # Asegurarse de que el comentario sea una cadena de texto
    comment = str(comment)
    
    # Tokenizar el comentario
    input_tokens = tokenizer(comment, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    input_tokens = {key: value.to(device) for key, value in input_tokens.items()}  # Mover los tokens a la GPU
    
    # Realizar la predicción
    with torch.no_grad():
        output = model(**input_tokens)
    
    predicted_label = torch.argmax(output.logits, dim=1).item()
    
    # Agregar la predicción a la lista
    predicted_labels.append(predicted_label)

# Agregar la lista de predicciones como una nueva columna al dataset
parte_data.loc[:, 'predicted_label'] = predicted_labels

# Guardar el nuevo dataset con las predicciones
parte_data.to_csv("D:\\clasificador_emociones\\steam_new_data_with_predictions.csv", index=False)
