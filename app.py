import streamlit as st
from app import LinearRegression
import pandas as pd
import numpy as np

upload = st.file_uploader('Upload', type=['csv'])

if upload is not None:
    
    # --- MINIMAL FIX: Run training ONLY ONCE on upload ---
    if 'model' not in st.session_state:
        df = pd.read_csv(upload)
        df = df.dropna(subset=['total_bedrooms'])
        
        model = LinearRegression()
        df = model.encode(df, 'ocean_proximity')
        train, test = model.test_train(df)

        xtrain = train.drop('median_house_value', axis=1).values
        ytrain = train['median_house_value'].values

        model.fit(xtrain, ytrain)
        
        # Save variables so they don't reset when you type a number
        st.session_state.model = model
        st.session_state.encoder_map = model.m['ocean_proximity']
        st.success('Model trained:')

    # --- Use the saved model and map from session state ---
    longitude = st.number_input('Longitude')
    latitude = st.number_input('latitude')
    housing_median_age = st.number_input('housign median age')
    total_rooms = st.number_input('total rooms')
    total_bedrooms = st.number_input('total bedrooms')
    population = st.number_input('population')
    households = st.number_input('households')
    median_income = st.number_input('median income')
    
    # --- MINIMAL FIX: Dynamically load categories from your encoder map ---
    categories = list(st.session_state.encoder_map.keys())
    ocean = st.selectbox('ocean', categories)
    encoded = st.session_state.encoder_map[ocean] 

    feature = [[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, encoded]]

    if st.button('predict'):
        # Call predict using the saved model state
        pred = st.session_state.model.predict(feature)
        st.write('prediction:', pred)