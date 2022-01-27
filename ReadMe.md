Project Description:
=====================
This project is an online platform to detect hate speech in an online document.


How to use:
==============
To use reproduce this platform, you have to follow these steps:

1- Download the data
=======================
The Data used in this project are:

1- Twitter dataset from "Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitte "

2- Wikipedia Talk Pages datasets from "Ex Machina: Personal Attacks Seen at Scale"

The datsets can be obtained by contacting the authhors. After hte datsets are downloaded, they should be placed in the following path "Data/Textual_Data" in the main project directory.


2- Download word embeddings
============================
In this project I use the glove common crawl to train LSTM and BiLSTM models. This can be downloaded from https://nlp.stanford.edu/projects/glove/
After that place the downloaded file in the following folder Data/word_embeddings/Glove

3- Traine Models
================
To train the used models, you have to install all the packages in requirements.txt. Then run bash files Train_model.sh after specifying the model, dataset, word emebddings to trian. The trianing process saves the trained models in trained_models.

make sure to train the models and that the paths to the trained models are correct to be able to run the demo.

4-Demo
=======
To run the demo, Make sure to change directory to the demo folder, then run the following command:

streamlit run demo.py

For the design to work, make sure that the settings of the web page set change to dark theme



