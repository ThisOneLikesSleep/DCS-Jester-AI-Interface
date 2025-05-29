import torch
import whisper
import time
import spacy
import csv
import openai
import json
import uvicorn
import socketio
import joblib
import websockets
import threading
import asyncio
import _queue
import torch.nn as nn
import numpy as np
from multiprocessing import Process, Queue
from sentence_transformers import SentenceTransformer
from classification_data import *
from send_command import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from udp_server import *

log_file = 'log.csv'

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def classify(queue, cuda_bool, stop_event, sock_j_outbound, sock_j_inbound, sock_d_inbound):
    print('Classify running.')
    print(cuda_bool.value)
    if cuda_bool.value:
        device = 'cuda'
    else:
        device = 'cpu'

    dict_data = {}
    for i in range(len(commands_index)):
        com = commands_index[i]
        list_com = commands[str(i)]
        dict_data[com] = list_com

    nlp = spacy.load("en_core_web_sm")

    transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    model = whisper.load_model('small.en', device=device)
    prompt = (f'You will hear a command given from an air force pilot.'
              f'Expect military aviation jargon.')

    normalizer = joblib.load('normalizer.pkl')
    pca = joblib.load('pca.pkl')
    cmd_model = SimpleNN(61, 32, 59, 0.3)
    cmd_model.load_state_dict(torch.load('commands.pth'))
    cmd_model.eval()

    with open(log_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "transcription", "corrected text",
                                                  "category", "time taken"])
        writer.writeheader()
        while not stop_event.is_set():
            audio = queue.get()
            start = time.time()
            text = model.transcribe(audio, initial_prompt=prompt)['text']
            print(text)

            # preprocess text
            doc = nlp(text)
            lemmatized_tokens = []

            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                # Preserve case for acronyms
                elif token.text.isupper() and len(token.text) > 1:
                    lemmatized_tokens.append(token.text)
                else:
                    lemmatized_tokens.append(token.lemma_.lower())

            lemmatized_sentence = " ".join(lemmatized_tokens)

            encoded_text = transformer_model.encode(lemmatized_sentence)
            encoded_text = normalizer.fit_transform(encoded_text.reshape(1, -1))
            encoded_text = pca.transform(encoded_text)
            encoded_text = torch.tensor(encoded_text)
            print(encoded_text)

            with torch.no_grad():
                prediction = cmd_model(encoded_text)

            predicted_class = int(torch.argmax(prediction, dim=1))

            print(predicted_class)

            send_command(sock_j_outbound, sock_j_inbound, sock_d_inbound,
                         predicted_class, nlp)

            end = time.time()
            print(f'Took {end - start:.2f} seconds to process.')

    websocket_process.join()
    return