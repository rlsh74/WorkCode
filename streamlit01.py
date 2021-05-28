# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:55:28 2021

@author: Шевченко Роман
"""

import streamlit as st

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()

with header:
    st.title('Welcome to data science project!')
    st.text('In this project Bla bla')

with dataset:
    st.header('NYC dataset')
    st.text('I found this dataset on blablabla.com')
    
with features:
    st.header('The features I created')
    
with modelTraining:
    st.header('Time to train the model')
    st.text('Here you get to choose hyperparameters of the model and see how the performance changes.')
    