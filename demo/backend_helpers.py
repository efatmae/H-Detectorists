from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import TextClassificationPipeline
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import frontend_helpers as fe
import pandas as pd
import numpy as np
import time
'''This python file contain functions that handle the backend side of the webservice.
 It connects the user interface to the trained ML models'''

def bert_predict(model, tokenizer, text):
    '''This function assesses the bert model against the text the user entered. The parameters are:
    Args:
        model: the saved trained BERT model
        tokenizer: BERT tokenizer used to convert the text into a format understandable by BERT
        text: the text from the user interface to test against the model - string
    Return:
        positive_label_probability: The model outcome which is a float represents a probability between 0 and 1 - float
    '''
    classification_pipeline = TextClassificationPipeline(model=model,
                                                                tokenizer=tokenizer,
                                                                framework='pt',
                                                                binary_output=True,
                                                                return_all_scores=True)
    positive_label_probability = classification_pipeline(text)[0][1]["score"]
    return positive_label_probability

def load_bert_model(model_path):
    '''This function loaded a saved trained BERT model. The parameters are:
    Args:
        model_path: the local path of hte saved trained BERT model - string
    Return:
        bert_tokenizer: the saved BERT tokenizer
        bert_model: the saved BERT model
    '''
    bert_tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    bert_model = BertForSequenceClassification.from_pretrained(model_path,
        # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False  # Whether the model returns all hidden-states.
    )

    return bert_tokenizer, bert_model

def load_rnn_model(model_path):
    '''This function loaded RNN model (LSTM and BiLSTM). The parameters are:
    Args:
        model_path: the path to the local saved RNN model - string
    Return:
        model: the saved RNN model

    '''
    model = tf.keras.models.load_model(model_path)
    return model

def rnn_predict(model, fit_on_text, sentence, maxlen):
    '''This function assesses the RNN model (LSTM adn BiLSTM) against the text the user entered. The parameters are:
    Args:
        model: the saved trained RNN model
        fit_on_text:  convert the text into a format understandable by RNN models
        sentence: the text from the user interface to test against the model - string
        maxlen: the length of sentence to be processed. ti changes with the used dataset - integer
    Return:
        model_prediciton: The model outcome which is a float represents a probability between 0 and 1 - float
    '''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(fit_on_text)
    X_test = tokenizer.texts_to_sequences([sentence])
    X_test = pad_sequences(X_test, maxlen=maxlen)
    model_prediciton = model.predict_proba(X_test)
    return model_prediciton[0][0]

def CB_detection_rnn(sexism_model,racism_model,aggression_model,sentence):
    '''This function uses RNN (LSTM and BiLSTM) models for sexism, racism amd aggression
     and assess the text from the user interface against the RNN models. The function parameters are:
     Args:
        sexism_model: the saved trained sexism model
        racism_model: the saved trained racism model
        aggression_model: the saved trained aggression model
        sentence: the sentence (string) to be assessed against each of sexism, racism, and aggression RNN models
    Return:
        sexism_prediction: the probability of the sentence contain sexism - float
        racism_prediction: the probability of the sentence contain racism - float
        aggression_prediction: the probability of the sentence contain aggression - float
     '''
    twitter_sexism_data = pd.read_csv("../Data/Textual_data/Twitter_sexism/Twitter_sex_data_train.csv").dropna()
    twitter_racism_data = pd.read_csv("../Data/Textual_data/Twitter_racism/Twitter_rac_data_train.csv").dropna()
    wtp_agg_data = pd.read_csv("../Data/Textual_data/wikipedia_aggression/wp_agg_data_train.csv").dropna()

    sexism_prediction = rnn_predict(sexism_model, twitter_sexism_data["Text_clean"], sentence, 21)
    racism_prediction = rnn_predict(racism_model, twitter_racism_data["Text_clean"], sentence, 21)
    aggression_prediction = rnn_predict(aggression_model, wtp_agg_data["Text_clean"], sentence, 1000)

    return sexism_prediction, racism_prediction, aggression_prediction

def CB_detection_bert(sexism_bert_tokenizer, sexism_bert_model,
                      racism_bert_tokenizer, racism_bert_model,
                      aggression_bert_tokenizer, aggression_bert_model,
                      sentence):
    '''This function uses BERT models for sexismn racism and aggression
    and assess the text from the user interface against the RNN models. The function parameters are:
     Args:
        sexism_bert_tokenizer: the tokenizer of the sexism BERT model
        sexism_model: the saved trained sexism BERT model
        racism_bert_tokenizer: the tokenizer of the racism BERT model
        racism_model: the saved trained racism BERT model
        aggression_bert_tokenizer:  the tokenizer of the aggression BERT model
        aggression_model: the saved trained aggression BERT model
        sentence: the sentence (string) to be assessed against each of sexism, racism, and aggression BERT models
    Return:
        sexism_prediction: the probability of the sentence contain sexism - float
        racism_prediction: the probability of the sentence contain racism - float
        aggression_prediction: the probability of the sentence contain aggression - float
        '''

    sexism_prediction = bert_predict(sexism_bert_model, sexism_bert_tokenizer, sentence)
    racism_prediction = bert_predict(racism_bert_model, racism_bert_tokenizer, sentence)
    aggression_prediction = bert_predict(aggression_bert_model, aggression_bert_tokenizer, sentence)

    return sexism_prediction, racism_prediction, aggression_prediction
