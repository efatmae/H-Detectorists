import streamlit as st
import backend_helpers as ws
import numpy as np

'''This python fie contain functions the formats and displays the model's outcomes to the user interface'''

def style_text(text, fontcolor, fontsize,fontweight, font_style="normal",indentation=True):
    '''This function styles a sentece to be displayed on the user interface. The parameters are:
    Args:
        text: a string of text to be styled
        fontcolor: a string with the color that we want the text to displayed in
        fontsize: a string of a number sets the font size we want the text to displayed in
        fontweight: a string of the font weight (normal or bold) we want the text to displayed in
        font_sty;e: a string of the font style (normal or italic) we want the text to displayed in
        indentation: a boolean that indicates if the sentence is print in the beginning of the sentence or after an indentation
    Returns:
        a string of the CSS style to display the text in on the interface
    '''
    if indentation == True:
        return '<p style="font-family:Courier;text-indent: 30px; color:'+ fontcolor + '; font-size:'+fontsize+'px;font-weight:'+ fontweight+'; font-style:'+ font_style +'">'+ text +'</p>'
    else:
        return '<p style="font-family:Courier; color:' + fontcolor + '; font-size:' + fontsize + 'px;font-weight:' + fontweight + '; font-style:' + font_style + '">' + text + '</p>'

def display_model_outcome(prediction_score, CB_type):
    '''This function displays the styled model outcome. The parametrs are:
    Args:
        prediction_score: the model prediction scores - float
        CB_type: the type of bullying (sexism, racism or aggression) - string
    Returns:
        styled_outcome: a string with the CSS styled models ' predictions
    '''
    if prediction_score > 0.5:
        if CB_type == "sexism":
           model_outcome = CB_type+' probability is ' + "{:.2f}".format(prediction_score*100) + "%"
           styled_outcome = style_text(model_outcome, "Orange", "20", "normal", "italic")
        elif CB_type == "racism":
           model_outcome = CB_type+' probabilit is ' + "{:.2f}".format(prediction_score*100) + "%"
           styled_outcome = style_text(model_outcome, "#53adcb", "20", "normal", "italic")
        elif CB_type == "aggression":
           model_outcome = CB_type + ' probability is ' + "{:.2f}".format(prediction_score*100) + "%"
           styled_outcome = style_text(model_outcome, "#d8c359", "20", "normal", "italic")
    else:
       model_outcome = CB_type+' probability is ' + "{:.2f}".format(prediction_score*100) + "%"
       styled_outcome = style_text(model_outcome, "Green", "20", "normal")
    return styled_outcome

def display_prediciton (sexism_prediction, racism_prediction, aggression_prediction):
    '''This function displays the model's prediction on the user interface. The parameters are:
    Args:
        sexism_prediction: a sexism model prediction - float
        racism_prediction: a sexism model prediction - float
        aggression_prediction: a sexism model prediction - float
    Returns:
        sexism_styles_output: a string with the CSS styled sexism model's predictions
        racism_styles_output: a string with the CSS styled racism model's predictions
        aggression_styles_output: a string with the CSS styled aggression model's predictions
    '''
    sexism_styles_output = display_model_outcome(sexism_prediction, "sexism")
    racism_styles_output = display_model_outcome(racism_prediction, "racism")
    aggression_styles_output = display_model_outcome(aggression_prediction, "aggression")

    return sexism_styles_output, racism_styles_output, aggression_styles_output
