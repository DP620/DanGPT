import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
os.system('cls') 

keras = tf.keras
layers = keras.layers
models = keras.models


import random
import json
import hashlib
import pickle

with open("intents.json", "rb") as f:
    file_content = f.read()
    current_hash = hashlib.md5(file_content).hexdigest()

data = json.loads(file_content)
try:
    with open("hash.txt", "r") as f:
        old_hash = f.read()
except FileNotFoundError:
    old_hash = ""

if current_hash != old_hash:
    print("Detected changes in 'intents.json' Resetting model...")
    if os.path.exists("data.pickle"):
        os.remove("data.pickle")
    if os.path.exists("model.keras"):
        os.remove("model.keras")
        
    with open("hash.txt", "w") as f:
        f.write(current_hash)
    
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            
            
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


model = models.Sequential([
    layers.Input(shape=(len(training[0]),)), 

    #Hidden Layers (The "Thinking" part)
    layers.Dense(10, activation='relu'), 
    layers.Dense(10, activation='relu'), 

    # Output Layer (The "Decision" part)
    layers.Dense(len(output[0]), activation='softmax')

])

# brain part to learn
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
try:
    model = models.load_model("model.keras")
    print("Model loaded from disk!")
except:
    print("No saved model found. Training now...")
    model.fit(training, output, epochs=1000, batch_size=8)
    model.save("model.keras")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return np.array(bag)           

def chat():
    print("start talking with the bot (type quit to stop) : ") 
    current_context = "" # Memory starts empty
    
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict(np.array([bag_of_words(inp, words)]), verbose=0)
        results_index = np.argmax(results)
        results_max = np.max(results)
        tag = labels[results_index]
    
        if results_max > 0.7:
            found_intent = None
            
            # Find the right intent based on Tag AND Context
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    # Check if this intent requires a specific context
                    if 'context_filter' in tg and tg['context_filter'] == current_context:
                        # Only use it if it matches our current memory
                        if tg['context_filter'] == current_context:
                            found_intent = tg
                            break
            
            
        if not found_intent:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    if 'context_filter' not in tg or tg['context_filter'] == "":
                        found_intent = tg
                        break
        if found_intent:
            
            current_context = found_intent.get('context_set', "")
            
            responses = found_intent['responses']
            print(f"DEBUG: Tag: {tag} | New Context: {current_context} | accuracy: {results_max}")
            print("\n" + "DanGPT: " + random.choice(responses) + "\n")
                
        else:
            print(f"DEBUG: Tag: {tag} | New Context: {current_context} | accuracy: {results_max}")
            print("\n" + "DanGPT: " + "I'm not sure what you mean in this context." + "\n")
                
        
chat()