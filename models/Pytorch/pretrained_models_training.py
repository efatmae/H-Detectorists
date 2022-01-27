import time
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
from pretrained_models_helpers import *
import gc

torch.cuda.empty_cache()
gc.collect()

''' this python file contains functions uses the functions from the pretrained_model_helpers.py 
to train the BERT model on the chosen datasets.'''
def set_device():
    '''this function checks If there's a GPU available to use otherwise use a CPU'''
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device

def train_model(model,train_dataset,test_dataset,text_field, labelfield,tokenizer, maxlen, test_size,
                batch_size, no_epoches, no_iterations, lr, eps,
                      saver_name, results_file):
    '''
    This function trains a BERT model on the chosen dataset
    Args:
        model: BERT model
        train_dataset: pandas dataframe that contain the training dataset
        test_dataset: pandas dataframe that contain the testing dataset
        text_field: the name of the text column in the training and testing dataset - string
        labelfield: the name of the label column in the training and testing dataset - string
        tokenizer: BERT tokenizer
        maxlen: the lenight of the maximum sentence in the dataset which depends on the dataset - integer
        test_size: the size of the validation dataset - float
        batch_size: the size of a single batch to train the BERT model - integer
        no_epoches: the number of times to train a BERT model on a single batch - integer
        no_iterations: the number of times to train and validates the model
        lr: the learning parameter of the ML model - float
        eps: the learning parameter of the ML model - float
        saver_name: the name of the model to be saved in - string
        results_file: the path to the text files with the model evaluation results - string

    Returns:
        a trained BERT model, and evaluation metrics text file

    '''
    AUC_scores = []
    F1_scores = []
    training_time = []
    micro_F1_scores = []
    macro_F1_scores = []
    device =set_device()

    sentences = train_dataset[text_field].values
    labels = [int(i) for i in train_dataset[labelfield].values]
    test_sentences = test_dataset[text_field].values
    test_labels = [int(i) for i in test_dataset[labelfield].values]

    optimizer = AdamW(model.parameters(),
                      lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=eps  # args.adam_epsilon  - default is 1e-8.
                      )
    total_steps = (len(sentences)-(len(sentences)*test_size))* no_epoches

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    input_ids, attention_masks, labels = data_tokenization(sentences, labels, tokenizer, maxlen)

    for i in range(no_iterations):
        start_time = time.time()
        train_dataloader, validation_dataloader = split_data_into_stratified_train_and_valid(input_ids, attention_masks, labels, batch_size, test_size)

        for epoch in range(no_epoches):
            # I believe the 'W' stands for 'Weight Decay fix"
            train_loss, train_logits = train(model, scheduler, optimizer, train_dataloader, device)
            valid_loss, valid_logits = validate(model, validation_dataloader, device)
        test_input_ids, test_attention_masks, test_labels = data_tokenization(test_sentences, test_labels, tokenizer, maxlen)
        test_data_loader = create_test_set_data_loader(test_input_ids, test_attention_masks, test_labels, batch_size)
        flat_predictions, flat_true_labels = test_model_performance(model, test_data_loader, device)
        AUC_scores.append(roc_auc_score(flat_true_labels, flat_predictions))
        F1_scores.append(f1_score(flat_true_labels, flat_predictions))
        macro_F1_scores.append(f1_score(flat_true_labels, flat_predictions, average="macro"))
        micro_F1_scores.append(f1_score(flat_true_labels, flat_predictions, average="micro"))
        exc_time = time.time() - start_time
        training_time.append(exc_time)
        print("End iteration" + str(i))
        torch.cuda.empty_cache()

    model.save_pretrained(saver_name)
    tokenizer.save_pretrained(saver_name)
    test_dataset["BERT_prediction"] = flat_predictions
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    print("Done!")
    return test_dataset

