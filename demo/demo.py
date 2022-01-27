import pandas as pd
import streamlit as st
import backend_helpers as ws
import frontend_helpers as fe
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import base64
from io import BytesIO,StringIO
import xlsxwriter
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import fitz
import docx2txt

# his python file builds the user interface and assess the user input against the trained machine learning models'''
def to_excel(df):
    '''this function exports the model results as an excel sheet. the in and out parameters are:
    IN:
        df: is the pandas dataframe that we want to convert to excel format
    OUT:
        processed_data: is the Bytes object of the data in the df
    '''

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index = True, sheet_name='Hate_report')
    writer.save()
    processed_data = output.getvalue()
    return processed_data
def set_page_title(title):
    '''This function sets the title of the browser. The parameters that hte function takes are:
    IN:
        title: a string containing the web browser tab title - string
    '''
    st.sidebar.markdown(unsafe_allow_html=True, body=f"""
        <iframe height=0 srcdoc="<script>
            const title = window.parent.document.querySelector('title') \

            const oldObserver = window.parent.titleObserver
            if (oldObserver) {{
                oldObserver.disconnect()
            }} \

            const newObserver = new MutationObserver(function(mutations) {{
                const target = mutations[0].target
                if (target.text !== '{title}') {{
                    target.text = '{title}'
                }}
            }}) \

            newObserver.observe(title, {{ childList: true }})
            window.parent.titleObserver = newObserver \

            title.text = '{title}'
        </script>" />
    """)
def get_table_download_link(df):
    '''Generates a link allowing the data in a given panda dataframe to be downloaded
    This function takes in and out the following parameters:
    IN:
        df: a pandas dataframe of the table to be downloaded
    OUT:
        a link to download the df in the format of excel sheet - string
    '''
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="results.xlsx">Export Excel File</a>' # decode b'abc' => abc
def highlight_col(column):
    '''this function highlights the table cells with values > 0.5. The function parameters are:
    IN:
        it takes column name to be highlighted - string
    OUT:
        it returns the higlighted values of the column with values > 0.5
    '''

    highlight = 'background-color: yellow;'
    default = ''
    return [highlight if v > 0.5 else default for v in column]
def extract_pdf_data(uploaded_pdf):
    '''This function reads a pdf file and returns the text as a string.The function parameters are:
    IN:
        uploaded_pdf : the uploaded file
    OUT:
        return a string with the textual content of the pdf - string

    '''
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.getText()
    doc.close()
    return text
def page_home():
    '''this function designs the left panel of the interface and connects it to the backend functions'''
    sexism_styled_output = []
    racism_styled_output = []
    aggression_styled_output = []

    sexism_predicitons = []
    racism_predictions = []
    aggression_predictions = []

    string_data = ""
    # Writing App Title and Description
    set_page_title("H-detectorists Demo v1")
    st.image('H-Detectorists_logo.png', width=300)
    st.title('Online Platform To Detect Hateful and Abusive Content')
    st.write('This app assesses textual documents for incidences of hateful and abusive content.')
    uploaded_file = st.file_uploader("Choose a file", type=["pdf","txt","docx"]) # file uploader

    selected_model = st.selectbox('Select a model:', ['--choose model--', 'Model I - BERT Based', 'Model II - BiLSTM Based']) # dropdownbox to select the model from
    predict_button = st.button('Assess') # the assess  button

    if uploaded_file: # when the user chooses a file to upload
        file_type = uploaded_file.type # check the extension of hte uploaded file
        if file_type == 'text/plain':
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8")) # read an uploaded textfile
            string_data = stringio.read()
        elif file_type == 'application/pdf':
            string_data = extract_pdf_data(uploaded_file) # read an uploaded pdf file
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
             string_data = docx2txt.process(uploaded_file) # read a Microsoft docx

    # prepare the text from the uplaoded file to be assessed against the selected ML models
    string_data = string_data.replace("\n", "") # remove new lines from the uploaded string
    sentences = string_data.split(".") # split the uploaded string into sentences based on "."
    sentences = sentences[:-1] # remove the last item after hte last "."
    sentences = [i for i in sentences if i != ""] # remove empty lines in the text.
    input = sentences
    if selected_model == "Model I - BERT Based": #if the user chose the BERT model
        # set the paths to the local trained 3 BERT models for sexism, racism and aggression
        sexism_model_path = "../trained_models/BERT-Fine-Tuned/Pytorch/Fine_Tune_Twitter_sexism_clean_text_64"
        racism_model_path = "../trained_models/BERT-Fine-Tuned/Pytorch/Fine_Tune_Twitter_racism_clean_text_64"
        aggression_model_path = "../trained_models/BERT-Fine-Tuned/Pytorch/Fine_Tune_wtp_agg_clean_text128"
        # load the 3 BERT models
        sexism_bert_tokenizer, sexism_bert_model = ws.load_bert_model(sexism_model_path)
        racism_bert_tokenizer, racism_bert_model = ws.load_bert_model(racism_model_path)
        aggression_bert_tokenizer, aggression_bert_model = ws.load_bert_model(aggression_model_path)
    elif selected_model =="Model II - BiLSTM Based":#if the user chose the Bi-LSTM model
        # set the paths to the local trained 3 Bi-LSTM models for sexism, racism and aggression
        sexism_model_path = "../trained_models/BiLSTM/Twitter_sexism_glove_cc_emb.h5"
        racism_model_path = "../trained_models/BiLSTM/Twitter_racism_glove_cc_emb.h5"
        aggression_model_path = "../trained_models/BiLSTM/wp_agg_keras_emb.h5"
        # load the 3 Bi-LSTM models
        sexism_model = ws.load_rnn_model(sexism_model_path)
        racism_model = ws.load_rnn_model(racism_model_path)
        aggression_model = ws.load_rnn_model(aggression_model_path)
    if predict_button:
        for sentence in sentences:
            # for every sentence in the document against the loaded model
               if selected_model == "Model I - BERT Based": # if the loaded model is BERT
                   # check the sentence against the sexism, racism, and aggression models
                   # then return the model prediciton results for the 3 models
                   sexism_prediction, racism_prediction, aggression_prediction = ws.CB_detection_bert(sexism_bert_tokenizer, sexism_bert_model,
                                     racism_bert_tokenizer, racism_bert_model,
                                     aggression_bert_tokenizer, aggression_bert_model, sentence)
                   # add the prediction scores of each sentence to a list
                   sexism_predicitons.append(sexism_prediction)
                   racism_predictions.append(racism_prediction)
                   aggression_predictions.append(aggression_prediction)
                   # display the model resutls for each sentece color coded based on the prediction score
                   sexism_styles_output, racism_styles_output, aggression_styles_output = fe.display_prediciton(sexism_prediction, racism_prediction, aggression_prediction)
                   sexism_styled_output.append(sexism_styles_output)
                   racism_styled_output.append(racism_styles_output)
                   aggression_styled_output.append(aggression_styles_output)

               elif selected_model == "Model II - BiLSTM Based":# if the loaded model is BiLSTM
                   # check the sentence against the sexism, racism, and aggression models
                   # then return the model prediciton results for the 3 models
                   sexism_prediction, racism_prediction, aggression_prediction = ws.CB_detection_rnn(sexism_model,racism_model,
                                    aggression_model, sentence)
                   sexism_predicitons.append(sexism_prediction)
                   racism_predictions.append(racism_prediction)
                   aggression_predictions.append(aggression_prediction)
                   # display the model resutls for each sentece color coded based on the prediction score
                   sexism_styles_output, racism_styles_output, aggression_styles_output = fe.display_prediciton(
                       sexism_prediction, racism_prediction, aggression_prediction)
                   sexism_styled_output.append(sexism_styles_output)
                   racism_styled_output.append(racism_styles_output)
                   aggression_styled_output.append(aggression_styles_output)


        sexism_tf = ["True" if i > 0.5 else "False" for i in sexism_predicitons] # get the sexist sentences when the sexism prediction > 0.5
        racism_tf = ["True" if i > 0.5 else "False" for i in racism_predictions]# get the racist sentences when the racism prediction > 0.5
        aggression_tf = ["True" if i > 0.5 else "False" for i in aggression_predictions]# get the aggressive sentences when the aggression prediction > 0.5

        sexism_prct = ["{:.3f}".format(i*100) for i in sexism_predicitons] # format the prediction score as a 3 floating points percentages
        racism_prct = ["{:.3f}".format(i*100) for i in racism_predictions] # format the prediction score as a 3 floating points percentages
        aggression_prct = ["{:.3f}".format(i*100) for i in aggression_predictions] # format the prediction score as a 3 floating points percentages

        sexism_score = ["{:.5f}".format(i) for i in sexism_predicitons] # format the prediction score as a 5 floating points
        racism_score = ["{:.5f}".format(i) for i in racism_predictions]# format the prediction score as a 5 floating points
        aggression_score = ["{:.5f}".format(i) for i in aggression_predictions]# format the prediction score as a 5 floating points

        #create the table with each sentence and the prediction scores of sexism, racism and aggression to be displayed
        results_df = pd.DataFrame({"Sentence": sentences,
                                   "Sexism": sexism_predicitons,
                                   "Racism": racism_predictions,
                                   "Aggression": aggression_predictions})
        #Create the detailes resutls table to be exported
        outcomes_df_header = pd.MultiIndex.from_product([['Sexism','Racism','Aggression'],
                                     ['T/F','probability']])
        rows_data = [sexism_tf, sexism_prct,
                    racism_tf, racism_prct,
                    aggression_tf, aggression_prct]
        s_data = list(map(list, zip(*rows_data)))
        outcomes_df = pd.DataFrame(s_data,columns=outcomes_df_header)
        outcomes_df.insert(0, 'Sentences', sentences)
        st.session_state.df = outcomes_df

        # add the model results to full results expandable
        with st.session_state["results_exp"]:
            for i in sentences:
                # set the style for each sentence in the document
                    styled_sentence = fe.style_text(i, "White", "20", "bold", "normal",False)
                # diaplsy the sentence using the set style
                    st.write(styled_sentence, unsafe_allow_html=True)
                    st.write(sexism_styled_output[input.index(i)], unsafe_allow_html=True)
                    st.write(racism_styled_output[input.index(i)], unsafe_allow_html=True)
                    st.write(aggression_styled_output[input.index(i)], unsafe_allow_html=True)
        # add teh results to the positive analysis expandable
        with st.session_state["analysis_exp"]:
            sex_pos_results_df = results_df[(results_df["Sexism"] > 0.5)] # get sexist results
            rac_pos_results_df = results_df[(results_df["Racism"] > 0.5)]# get racist results
            agg_pos_results_df = results_df[(results_df["Aggression"] > 0.5)]# get aggressive results
            # prepare the sexism pie chart
            sexism_piechart_data = [len(results_df[results_df['Sexism'] > 0.5]) * 100 / len(results_df),
                                   len(results_df[results_df['Sexism'] < 0.5]) * 100 / len(results_df)]
            # prepare the racism pie chart
            racism_piechart_data = [len(results_df[results_df['Racism'] > 0.5]) * 100/ len(results_df),
                                    len(results_df[results_df['Racism'] < 0.5]) * 100/ len(results_df)]
            # prepare the aggression pie chart
            aggression_piechart_data = [len(results_df[results_df['Aggression'] > 0.5]) * 100/ len(results_df),
                                    len(results_df[results_df['Aggression'] < 0.5]) * 100/ len(results_df)]
            # prepare the pie charts legend labels
            sex_labels = 'Sexist', 'Normal'
            rac_labels = 'Racist', 'Normal'
            agg_labels = 'Aggressive', 'Normal'
            agg_sentences = ["<br>"+ i+ "</br>" for i in agg_pos_results_df.Sentence.values] # get the aggreeive sentences and put them in a list with new line between them
            sex_sentences = ["<br>"+ i+ "</br>" for i in sex_pos_results_df.Sentence.values] # get the sexist sentences and put them in a list with new line between them
            rac_sentences = ["<br>"+ i+ "</br>" for i in rac_pos_results_df.Sentence.values] # get the racist sentences and put them in a list with new line between them

            # settings of the pie charts
            specs = [[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]]
            layout = go.Layout(
                legend=dict(
                    orientation="h")
            )
            font = dict(
                family="Courier New, monospace",
                size=18)
            hoverlabel = dict(
                font_size=12,
                font_family="Rockwell"
            )
            # create the pie charts
            fig = make_subplots(rows=1, cols=3, specs=specs, subplot_titles=('Sexism','Racism','Aggression'))
            fig.add_trace(go.Pie(labels=sex_labels, values=sexism_piechart_data,
                                 text=[sex_sentences,""], marker_colors=["orange", "green"]),1,1)

            fig.add_trace(go.Pie(labels=rac_labels, values=racism_piechart_data,
                                 text=[rac_sentences,""], marker_colors=["#53adcb", "green"]),1,2)

            fig.add_trace(go.Pie(labels=agg_labels, values=aggression_piechart_data,
                                         text=[agg_sentences,""], marker_colors=["#d8c359", "green"]),1,3)
            fig.update_traces(hoverinfo='text', textinfo='percent')
            fig.update_layout(layout,width=650, height=500, font=font,hoverlabel=hoverlabel)
            st.plotly_chart(fig)

            #results_df.style.set_properties(subset=['Sentence'], **{'width': '3000px'})
            #st.dataframe(results_df.style.apply(highlight_col, subset=["Sexism", "Racism", "Aggression"], axis=0))

        # prepare the export data
        with st.session_state["Download_exp"]:
            st.markdown(get_table_download_link(st.session_state.df), unsafe_allow_html=True)
def main():
    # create hte expanders on the right side of the user interface
    st.session_state["analysis_exp"] = st.expander("Positive Results Analysis", expanded=True)
    st.session_state["results_exp"] = st.expander("Full Model Results", expanded=True)
    # And within an expander
    st.session_state["Download_exp"] = st.expander("Export Results", expanded=True)

    # AND in st.sidebar!
    with st.sidebar:
        page_home()
if __name__ == "__main__":
    main()
