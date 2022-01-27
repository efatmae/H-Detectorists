import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import contractions_dict
from collections import Counter
from string import punctuation
from nltk.tokenize import word_tokenize

'''helper functions that are are used in training the existing models and to train new models'''

def noise_cleaning_preprocesing(text, remove_twitter_rev, remove_qoute, remove_stopwords, remove_punctuation):
    '''
    This function preprocess the textual dataset before training a ML model on
    Args:
        text: the text to be assessed against the ML model - string
        remove_twitter_rev: indicator to remove twitter twitter handles from the text - boolean
        remove_qoute: indicator to remove single and double qoutes  from the text - boolean
        remove_stopwords: indicator to remove stop words from the text - boolean
        remove_punctuation: indicator to remove the punctuation from the text - boolean

    Returns:
        text: the preprocessed text - string
    '''
    import preprocessor as p
    from contractions_dict import CONTRACTION_MAP
    import unicodedata
    from nltk.corpus import stopwords
    import re

    if remove_twitter_rev == True:
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED)
    else:
        p.set_options(p.OPT.URL, p.OPT.MENTION)

    def camel_case_split(identifier):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    def remove_tags(text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    def custome_remove_punctuation(words):
        import string
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        return stripped

    def remove_unwanted_signs(text):
        if remove_qoute == True:
            text = text.replace('"', "")
        # text = text.replace('\\', "")
        # text = text.replace('!', "")
        # text = text.replace('/', "")
        # text = text.replace('*', "")
        text = text.replace("\n", "")
        # text = text.replace(":","")
        text = text.replace("#", "")
        text = text.replace("&amp", "and")
        text = text.replace("&lt", "<")
        text = text.replace("&gt", ">")
        return text

    def custome_remove_stop_words(words):
        keep_words = ["you", "your", "yours", "he", "him", "his", "she", "her", "hers", "they", "them", "their",
                      "theirs"]
        stop_words = set(stopwords.words('english'))
        stop_words = [word for word in stop_words if not word in keep_words]
        words = [w for w in words if not w in stop_words]
        return words

    clean_text = remove_unwanted_signs(text)
    clean_text = " ".join(camel_case_split(clean_text))
    encoded_string = clean_text.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    clean_text = remove_tags(decode_string)
    clean_text = remove_accented_chars(clean_text)
    clean_text = p.clean(clean_text)
    clean_text = expand_contractions(clean_text)

    tokens = word_tokenize(clean_text)
    words = [word.lower() for word in tokens]
    words = [word for word in words if not word.isdigit()]  # remove numbers

    if remove_stopwords == True:
        words = custome_remove_stop_words(words)

    if remove_punctuation == True:
        words = custome_remove_punctuation(words)

    text = " ".join(words)
    text = re.sub(' +', ' ', text)
    return text
def data_tokenization(sentences,labels,tokenizer, maxlen):
    '''
    Tokenize all of the sentences and map the tokens to thier word IDs
    Args:
        sentences: the sentences in the dataset to be assessed the ML against - list
        labels: the labels of these sentences in the dataset - list
        tokenizer: BERT tokenizer
        maxlen: the maximum length of a sentence in the dataset - integer

    Returns:
            parameters used by the BERT model in the training process
    '''
    input_ids = []
    attention_masks = []
    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=maxlen,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',
            truncation='longest_first'  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels
def split_data_into_stratified_train_and_valid(input_ids, attention_masks, labels, batch_size=32, test_size=0.3):
    '''
    This function split the training dataset into training and validation sets
    Args:
        input_ids, attention_masks, labels: outputs of the data_tokenization function
        batch_size: the size of batch dataset to train dataset in one itertion- default value = 32 - integer
        test_size: the percentage of the validation set = default = 0.3 - float
    Returns:
            train_dataloader, validation_dataloader: split datasets to be used in the training process
    '''
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_idx, valid_idx = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        shuffle=True,
        stratify=labels)
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(train_idx),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        dataset,  # The validation samples.
        sampler=SequentialSampler(valid_idx),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader
def train(model, scheduler, optimizer, train_dataloader, device):
    '''
    This function trians a bERT model using pytorch platform
    Args:
        model: the BERT model
        scheduler: parameters needed for training pytorch BERT model
        optimizer: parameters needed for training pytorch BERT model
        train_dataloader: the split training dataset
        device: parameters needed for training pytorch BERT model

    Returns: model predictions on the training dataset

    '''
    print('Training...')
    model.train()
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        loss, logits,_, _ = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return loss, logits
def validate(model, validation_dataloader, device):
    '''
    This function validates a bERT model using pytorch platform
    Args:
        model: the BERT model
        validation_dataloader: the split validation dataset
        device: parameters needed for training pytorch BERT model

    Returns: model predictions for the validation dataset

    '''
    model.eval()
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            loss, logits,_,_ = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
    return loss, logits
def create_test_set_data_loader(input_ids, attention_masks, labels, batch_size=32):
    '''
    This function prepares the test dataset to evaluate the BERT model
    Args:
        input_ids, attention_masks,labels: outputs of the data_tokenization function
        batch_size: the size of batch dataset to train dataset in one itertion- default value = 32 - integer

    Returns:
            prediction_dataloader: the test dataset
    '''
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    return prediction_dataloader
def test_model_performance(model, test_data_loader, device):
    '''
    This function evaluates the trained model on the test dataset
    Args:
        model: the trained BERT model
        test_data_loader: the test data ready to be used with the pytorch platform
        device:

    Returns:
            flat_predictions: the model prediction of the test set dataset
             flat_true_labels: the true labels of the test set.

    '''
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in test_data_loader:
        batch = tuple(t.to(device) for t in batch)  # Add batch to GPU

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)  # Forward pass, calculate logit predictions
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    return flat_predictions, flat_true_labels