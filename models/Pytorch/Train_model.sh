export TRAIN_FILE_PATH=Twitter_racism/Twitter_rac_data_train.csv
export TEST_FILE_PATH=Twitter_racism/Twitter_rac_data_test.csv
export TEXT_COL_NAME=Text_clean
export LABEL_COL_NAME=oh_label
export PREPROCESS_DATA=Yes
export RESULTS_FILE_NAME=Twitter_racism_results
export SAVED_MODEL_NAME=BERT_twitter_racism_model


python3 ./Trains_model.py \
--training_ds_file_path ${TRAIN_FILE_PATH} \
--test_ds_file_path ${TEST_FILE_PATH} \
--text_col_name ${TEXT_COL_NAME} \
--label_col_name ${LABEL_COL_NAME} \
--preprocess_data ${PREPROCESS_DATA} \
--results_file_name ${RESULTS_FILE_NAME} \
--saved_model_name ${SAVED_MODEL_NAME}
