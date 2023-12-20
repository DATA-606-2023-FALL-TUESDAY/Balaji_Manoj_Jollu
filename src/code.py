#!/usr/bin/env python
# coding: utf-8

# # DATA606 - Capstone Project - Property price prediction (Web Application Code )
# #### Author Name - Balaji Manoj Jollu
# #### Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# dataset : https://www.kaggle.com/code/goyaladi/property-price-ann-predictions/input

# In[2]:


file_path = 'Makaan_Properties_Buy.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')
df.head()


# In[3]:


df.dropna(inplace=True)


# In[4]:


df['Price_per_unit_area'] = df['Price_per_unit_area'].str.replace(',', '', regex=True).astype(float)


# In[5]:


df['Price'] = df['Price'].str.replace(',', '', regex=True).astype(float)


# In[6]:


df['Size'] = df['Size'].str.replace(' sq ft', '').str.replace(',', '').astype(int)


# In[7]:


df['Price_USD'] = (df['Price'] / 83).astype(int)


# In[8]:


df_Builder = df[['builder_id', 'Builder_name']].drop_duplicates()


# In[9]:


df_city = df[['City_id', 'City_name']].drop_duplicates()


# In[10]:


columns_to_drop = ['Property_Name','Property_id','Posted_On','Project_URL','description','Listing_Category','is_commercial_Listing','builder_id','Sub_urban_ID', 'Sub_urban_name','Price']
df = df.drop(columns=columns_to_drop)


# In[11]:


df_Property_type = pd.DataFrame({
    'Property_type': ['Apartment', 'Residential Plot', 'Villa', 'Independent Floor', 'Independent House'],
    'Mapped_value': [0, 1, 2, 3, 4]
})


df['Property_type'] = df['Property_type'].replace(df_Property_type.set_index('Property_type')['Mapped_value'])


# In[12]:


df_Property_status = pd.DataFrame({
    'Property_status': ['Under Construction', 'Ready to move'],
    'Mapped_value': [0, 1]
})


df['Property_status'] = df['Property_status'].replace(df_Property_status.set_index('Property_status')['Mapped_value'])


# In[13]:


df_Property_building_status = pd.DataFrame({
    'Property_building_status': ['UNVERIFIED', 'ACTIVE', 'INACTIVE'],
    'Mapped_value': [0, 1, 2]
})


df['Property_building_status'] = df['Property_building_status'].replace(df_Property_building_status.set_index('Property_building_status')['Mapped_value'])


# In[14]:


df_is_furnished = pd.DataFrame({
    'is_furnished': ['Unfurnished', 'Semi-Furnished', 'Furnished'],
    'Mapped_value': [0, 1, 2]
})


df['is_furnished'] = df['is_furnished'].replace(df_is_furnished.set_index('is_furnished')['Mapped_value'])


# In[15]:


df_True_False = pd.DataFrame({
    'True_False': [False, True],
    'Mapped_value': [0, 1]
})


df['is_plot'] = df['is_plot'].replace(df_True_False.set_index('True_False')['Mapped_value'])
df['is_RERA_registered'] = df['is_RERA_registered'].replace(df_True_False.set_index('True_False')['Mapped_value'])
df['is_Apartment'] = df['is_Apartment'].replace(df_True_False.set_index('True_False')['Mapped_value'])
df['is_ready_to_move'] = df['is_ready_to_move'].replace(df_True_False.set_index('True_False')['Mapped_value'])
df['is_PentaHouse'] = df['is_PentaHouse'].replace(df_True_False.set_index('True_False')['Mapped_value'])
df['is_studio'] = df['is_studio'].replace(df_True_False.set_index('True_False')['Mapped_value'])


# In[16]:


df_No_of_BHK = pd.DataFrame({
    'No_of_BHK': ['0 BHK','1 BHK','2 BHK','3 BHK','4 BHK','5 BHK','6 BHK','7 BHK','8 BHK','9 BHK','10 BHK', '11 BHK', '12 BHK', '14 BHK', '1 RK', '2 RK', '3 RK'],
    'Mapped_Value': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
})

# Replace values in df with values from mapping_df
df['No_of_BHK'] = df['No_of_BHK'].map(df_No_of_BHK.set_index('No_of_BHK')['Mapped_Value'])


# In[17]:


df.columns.tolist()


# In[18]:


df = df.drop(columns=['Builder_name','City_name'
                      ,'Locality_ID','Locality_Name','Longitude', 'Latitude',
                      'listing_domain_score'#,'Price_per_unit_area','Size'
                      ,'is_plot','is_RERA_registered','is_Apartment','is_ready_to_move','is_PentaHouse','is_studio'])




# In[19]:


df.columns.tolist()


# ##  Code for Predicting Values 

# In[20]:


X = df.drop(columns=['Price_USD'])
y = df['Price_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:



rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


# In[22]:


st.write('# Property Price Prediction')
st.write('#### DATA606 - Capstone Project' )
st.write('#### Author Name - Balaji Manoj Jollu')
st.write('#### Prepared for UMBC Data Science Master Degree Capstone')
st.write('#### Under Guidance of  Dr Chaojie (Jay) Wang)')
         


# In[23]:


# Streamlit UI
city_id_selected = st.selectbox('Select city', df_city['City_name'])
property_type_selected = st.selectbox('Select a type of Property', df_Property_type['Property_type'].unique())
No_of_BHK_selected = st.selectbox('Select Number of Bed Rooms ', df_No_of_BHK['No_of_BHK'].unique())
is_furnished_selected = st.selectbox('Select furnished Type  ', df_is_furnished['is_furnished'].unique())
property_status_selected = st.selectbox('Select status of Property', df_Property_status['Property_status'].unique())
property_building_status_selected = st.selectbox('Select Property Verification Status ', df_Property_building_status['Property_building_status'].unique())
with st.expander("Advanced Options"):
    Size_selected = st.slider('Select Size of property', min_value=df['Size'].min(),
                                 max_value=df['Size'].max(),
                                 value=(df['Size'].min(), df['Size'].max()))
    Price_per_unit_area_selected = st.slider('Select Price Range', min_value=df['Price_per_unit_area'].min(),
                                max_value=df['Price_per_unit_area'].max(),
                                value=(df['Price_per_unit_area'].min(), df['Price_per_unit_area'].max()))

# Filter DataFrames based on selected values
property_type = df_Property_type[df_Property_type['Property_type'] == property_type_selected]['Mapped_value'].values[0] if property_type_selected else df_Property_type['Mapped_value'].values[0]
property_status = df_Property_status[df_Property_status['Property_status'] == property_status_selected]['Mapped_value'].values[0] if property_status_selected else df_Property_status['Mapped_value'].values[0]
property_building_status = df_Property_building_status[df_Property_building_status['Property_building_status'] == property_building_status_selected]['Mapped_value'].values[0] if property_building_status_selected else df_Property_building_status['Mapped_value'].values[0]
city_id = df_city[df_city['City_name'] == city_id_selected]['City_id'].values[0] if city_id_selected else df_city['Mapped_value'].values[0]
No_of_BHK = df_No_of_BHK[df_No_of_BHK['No_of_BHK'] == No_of_BHK_selected]['Mapped_Value'].values[0] if No_of_BHK_selected else df_No_of_BHK['Mapped_Value'].values[0]
is_furnished_values = df_True_False[df_True_False['True_False'] == is_furnished_selected]['Mapped_value'].values
is_furnished = is_furnished_values[0] if is_furnished_values else df_True_False['Mapped_value'].values[0]


# Create a feature input dictionary for prediction
input_data_1 = {
    'Property_type': property_type,
    'Property_status': property_status,
    'Property_building_status': property_building_status,
    'City_id': city_id,
    'No_of_BHK' : No_of_BHK,
    'is_furnished': is_furnished,
    'Size' : Size_selected[0],  
    'Price_per_unit_area': Price_per_unit_area_selected[0],  
}

input_data_2 = {
    'Property_type': property_type,
    'Property_status': property_status,
    'Property_building_status': property_building_status,
    'City_id': city_id,
    'No_of_BHK' : No_of_BHK,
    'is_furnished': is_furnished,
    'Size' : Size_selected[1], 
    'Price_per_unit_area': Price_per_unit_area_selected[1],  
}


# In[24]:


if st.button("Run Prediction"):
    prediction_1 = int(rf_model.predict([list(input_data_1.values())]))
    prediction_2 = int(rf_model.predict([list(input_data_2.values())]))

    # Format predictions with commas
    formatted_prediction_1 = '{:,}'.format(prediction_1)
    formatted_prediction_2 = '{:,}'.format(prediction_2)

    st.write('### Prediction Price : $' + formatted_prediction_1 + ' - ' + formatted_prediction_2)

