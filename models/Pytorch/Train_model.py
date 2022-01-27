from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from pretrained_models_training import *
from pretrained_models_helpers import *
import pandas as pd
import argparse
import sys

# the function being called when the python file is executed
def main():
    # set the parameters to be sent by the users in the bash file with their properties.
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_ds_file_path", default=None, type=str, required=True,
                        help=" the path of the training dataset .csv file in the Data folder")
    parser.add_argument("--test_ds_file_path", default=None, type=str, required=True,
                        help="the path of the test dataset .csv file in the Data folder")
    parser.add_argument("--text_col_name", default=None, type=str, required=True,
                        help="the name of the textual column in the csv file")
    parser.add_argument("--label_col_name", default=None, type=str, required=True,
                        help="the name of the label column in the csv file")
    parser.add_argument("--preprocess_data", default=None, type=str, required=True,
                        help="an indicator of preprocess the data before training the model or not")
    parser.add_argument("--results_file_name", default=None, type=str, required=True,
                        help="the name of the file which has the model's evaluation metrics in the Results folder")
    parser.add_argument("--saved_model_name", default=None, type=str, required=True,
                        help="the name of the saved trained model saved in trained_models folder ")

    # reading the parameters that hte user send in the bash file
    args = parser.parse_args()
    training_ds_file_path = args.training_ds_file_path
    test_ds_file_path = args.test_ds_file_path
    text_col_name = args.text_col_name
    label_col_name = args.label_col_name
    preprocess_data = args.preprocess_data
    results_file_name = args.results_file_name
    saved_model_name = args.saved_model_name

    # Read datasets from the paths set by the user
    train_df = pd.read_csv("../../Data/Textual_data/"+training_ds_file_path, index_col=False)
    test_df = pd.read_csv("../../Data/Textual_data/"+test_ds_file_path, index_col=False)
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    #Model parameters
    batch_size = 32
    no_epochs = 10
    no_iterations = 5
    maxlen = 32
    lr = 2e-5
    eps = 1e-8
    saver_name = "../../trained_models/BERT-Fine-Tuned/Pytorch/"+saved_model_name # set the training model path
    results_files = "../Results/BERT_Fine_Tuned/"+results_file_name # set teh results file path

    if preprocess_data == "Yes":
        # Pre-process the data to be ready for training the model
        train_df[text_col_name] = train_df[text_col_name].apply(
            lambda x: noise_cleaning_preprocesing(x, remove_twitter_rev=False, remove_qoute=True,
                                                  remove_stopwords=False, remove_punctuation=False))
        test_df[text_col_name] = test_df[text_col_name].apply(
            lambda x: noise_cleaning_preprocesing(x, remove_twitter_rev=False, remove_qoute=True,
                                                  remove_stopwords=False, remove_punctuation=False))

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('../../trained_models/bert-base-uncased', do_lower_case=True)

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "../../trained_models/bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=True,  # Whether the model returns attentions weights.
        output_hidden_states=True,  # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()
     # Train a BERT model on teh dataset and word embeddings specified by the user
    train_model(model, train_df, test_df, text_col_name, label_col_name, tokenizer, maxlen, 0.3,
                                    batch_size, no_epochs, no_iterations, lr, eps,
                                    saver_name, results_files)

if __name__ == "__main__":
    main()