#!/usr/bin/env python
# coding: utf-8

# # DATA606 - Capstone Project - EDA
# ## Proposal Title: Property price prediction
# ### Author Name - Balaji Manoj Jollu
# ### Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import folium
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier


# In[2]:


st.write('# Property Price Prediction')
st.write('#### DATA606 - Capstone Project - EDA' )
st.write('#### Author Name - Balaji Manoj Jollu')
st.write('#### Prepared for UMBC Data Science Master Degree Capstone')
st.write('#### Under Guidance of  Dr Chaojie (Jay) Wang)')
         


# dataset : https://www.kaggle.com/code/goyaladi/property-price-ann-predictions/input

# In[3]:


st.write('dataset : https://www.kaggle.com/code/goyaladi/property-price-ann-predictions/input')


# In[4]:


file_path = 'Makaan_Properties_Buy.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')
df.head()


# In[5]:


st.write("## Displaying Head of DataFrame")
st.dataframe(df.head())


# ## Exploring dataset

# In[6]:


#df.info()


# In[7]:


st.write("## Size of Raw Data Set")
print(df.dtypes)
st.write(f"The size of the raw data set is: {df.shape}")


# In[8]:


st.write("## Column Names with Data Types")
st.write(df.dtypes)
print(df.dtypes)


# ## Data Cleaning

# In[9]:


df.dropna(inplace=True)


# In[10]:


print(df.isnull().sum())


# In[11]:


st.write("### Size of Data After Data Cleaning")
st.write(f"The size of the data after data cleaning is: {df.shape}")
print(df.shape)


# In[12]:


df['Price_per_unit_area'] = df['Price_per_unit_area'].str.replace(',', '', regex=True).astype(float)


# In[13]:


df['Price'] = df['Price'].str.replace(',', '', regex=True).astype(float)


# In[14]:


df['Size'] = df['Size'].str.replace(' sq ft', '').str.replace(',', '').astype(int)


# In[15]:


df['Price_USD'] = (df['Price'] / 83).astype(int)


# In[16]:


print(df.columns.tolist())


# In[17]:


df.head()


# In[18]:


df_Builder = df[['builder_id', 'Builder_name']].drop_duplicates()


# In[19]:


df_city = df[['City_id', 'City_name']].drop_duplicates()


# In[20]:


columns_to_drop = ['Property_Name','Property_id','Posted_On','Project_URL','description','Listing_Category','is_commercial_Listing','builder_id','Sub_urban_ID', 'Sub_urban_name','Price']
df = df.drop(columns=columns_to_drop)


# ## Visualizing
# the dependency of each column on the "price" column

# ### Below Graph shows the count of property type for each city in our data

# In[21]:


st.write('## Count of property type for each city ')


# In[22]:


fig, ax = plt.subplots(figsize=(14, 14))
sns.set(style="whitegrid")
ax = sns.countplot(data=df, y="City_name", hue="Property_type", order=df["City_name"].value_counts().index)
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f'{int(width)}', (width, p.get_y() + p.get_height() / 2.), va='center')
plt.title("Number of Property Type by City")
plt.xlabel("Count")
plt.ylabel("City Name")
plt.legend(title="Property Type")
st.pyplot(fig)


# ###  Average Price (USD) by Property Type and City

# In[23]:


st.write('## Average Price (USD) by Property Type and City')


# In[24]:


average_prices = df.groupby(['Property_type', 'City_name'])['Price_USD'].mean().reset_index()
average_prices = average_prices.sort_values(by='Price_USD', ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
sns.set(style="whitegrid")
sns.barplot(data=average_prices, x="Property_type", y="Price_USD", hue="City_name", ax=ax)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
plt.title("Average Price (USD) by Property Type and City (Sorted)")
plt.xlabel("Property Type")
plt.ylabel("Average Price (USD)")
plt.legend(title="City Name", loc="upper right")
st.pyplot(fig)


# ### Top Highest and Lowest Priced Properties on Map - Not working in Steamlit

# In[25]:


#st.write('## Top Highest and Lowest Priced Properties on Map')


# In[26]:


#top_highest = df.groupby('City_name').apply(lambda x: x.nlargest(50, 'Price_USD')).reset_index(drop=True)
#top_lowest = df.groupby('City_name').apply(lambda x: x.nsmallest(50, 'Price_USD')).reset_index(drop=True)
#concatenated_df = pd.concat([top_highest, top_lowest])
#concatenated_df.reset_index(drop=True, inplace=True)
#concatenated_df.shape
#m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
#for index, row in concatenated_df.iterrows():
#    latitude, longitude, price, city, location, bhk, property_type = row['Latitude'], row['Longitude'], row['Price_USD'], row['City_name'], row['Locality_Name'], row['No_of_BHK'], row['Property_type']
#    color = 'red' if index < 400 else 'blue'
#    popup_text = f"City: {city}<br>Location: {location}<br>Property Type: {property_type}<br>BHK: {bhk}<br>Price in USD: ${price}"
#    folium.Marker(
#        location=[latitude, longitude],
#        popup=folium.Popup(popup_text, max_width=300),
#        icon=folium.Icon(color=color)
#    ).add_to(m)
#
#m


# In[27]:



#top_highest = df.groupby('City_name').apply(lambda x: x.nlargest(50, 'Price_USD')).reset_index(drop=True)
#top_lowest = df.groupby('City_name').apply(lambda x: x.nsmallest(50, 'Price_USD')).reset_index(drop=True)
#concatenated_df = pd.concat([top_highest, top_lowest])
#concatenated_df.reset_index(drop=True, inplace=True)
#m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
#for index, row in concatenated_df.iterrows():
#    latitude, longitude, price, city, location, bhk, property_type = row['Latitude'], row['Longitude'], row['Price_USD'], row['City_name'], row['Locality_Name'], row['No_of_BHK'], row['Property_type']
#    color = 'red' if index < 400 else 'blue'
#    popup_text = f"City: {city}<br>Location: {location}<br>Property Type: {property_type}<br>BHK: {bhk}<br>Price in USD: ${price}"
#    folium.Marker(
#        location=[latitude, longitude],
#        popup=folium.Popup(popup_text, max_width=300),
#        icon=folium.Icon(color=color)
#    ).add_to(m)
#
#    
#st.write(m)    


# ### Histogram of Price per Unit Area

# In[28]:



st.write('## Histogram of Price per Unit Area')
fig, ax = plt.subplots()
df['Price_per_unit_area'].hist(ax=ax)
plt.title('Histogram of Price per Unit Area')
plt.xlabel('Price per Unit Area')
plt.ylabel('Frequency')
st.pyplot(fig)


# ### Histogram of Number of BHK

# In[29]:


st.write('## Histogram of Number of BHK')
fig, ax = plt.subplots()
df['No_of_BHK'].hist(ax=ax)
plt.title('Histogram of Number of BHK')
plt.xlabel('Number of BHK')
plt.ylabel('Frequency')
fig.set_size_inches(12, 10)
st.pyplot(fig)


# ### Top 10 Builders by Count

# In[30]:


st.write('## Top 10 Builders by Count')
builder_counts = df['Builder_name'].value_counts()[:10]
fig, ax = plt.subplots()
builder_counts.plot(kind='bar', ax=ax)
plt.title('Top 10 Builders by Count')
plt.xlabel('Builder Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)


# ### Average Price by Property Attributes

# In[31]:


st.write('## Average Price by Property Attributes')


for i in ['Property_type', 'Property_status', 'City_name', 'No_of_BHK', 'is_furnished', 'is_plot',
          'is_RERA_registered', 'is_Apartment', 'is_ready_to_move', 'is_PentaHouse', 'is_studio']:
    st.write(f"## Average Price by {i}")
    avg_prices = df.groupby(i)['Price_USD'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 8))
    avg_prices.plot(kind='barh', x=i, y='Price_USD', ax=ax)
    plt.title(f'Average Price by {i}')
    plt.xlabel('Average Price_USD')
    plt.ylabel(i)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    st.pyplot(fig)


# In[ ]:





# ## Mapping Values
# 

# In[32]:


df['Property_type'].value_counts()


# In[33]:


df_Property_type = pd.DataFrame({
    'Property_type': ['Apartment', 'Residential Plot', 'Villa', 'Independent Floor', 'Independent House'],
    'Mapped_value': [0, 1, 2, 3, 4]
})


df['Property_type'] = df['Property_type'].replace(df_Property_type.set_index('Property_type')['Mapped_value'])


# In[34]:


df['Property_status'].value_counts()


# In[35]:


df_Property_status = pd.DataFrame({
    'Property_status': ['Under Construction', 'Ready to move'],
    'Mapped_value': [0, 1]
})


df['Property_status'] = df['Property_status'].replace(df_Property_status.set_index('Property_status')['Mapped_value'])


# In[36]:


df['Property_building_status'].value_counts()


# In[37]:


df_Property_building_status = pd.DataFrame({
    'Property_building_status': ['UNVERIFIED', 'ACTIVE', 'INACTIVE'],
    'Mapped_value': [0, 1, 2]
})


df['Property_building_status'] = df['Property_building_status'].replace(df_Property_building_status.set_index('Property_building_status')['Mapped_value'])


# In[38]:


df['is_furnished'].value_counts()


# In[39]:


df_is_furnished = pd.DataFrame({
    'is_furnished': ['Unfurnished', 'Semi-Furnished', 'Furnished'],
    'Mapped_value': [0, 1, 2]
})


df['is_furnished'] = df['is_furnished'].replace(df_is_furnished.set_index('is_furnished')['Mapped_value'])


# In[40]:


df['is_plot'].value_counts()


# In[41]:


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


# In[42]:


df_True_False[df_True_False['True_False'] == True]['Mapped_value'].values[0]


# In[43]:


#df_True_False


# In[44]:


df['is_RERA_registered'].value_counts()


# In[45]:


df['is_Apartment'].value_counts()


# In[46]:


df[ 'is_ready_to_move'].value_counts()


# In[47]:


df['is_PentaHouse'].value_counts()


# In[48]:


df['is_studio'].value_counts()


# In[49]:


df['No_of_BHK'].value_counts()


# In[50]:


df_No_of_BHK = pd.DataFrame({
    'No_of_BHK': ['0 BHK','1 BHK','2 BHK','3 BHK','4 BHK','5 BHK','6 BHK','7 BHK','8 BHK','9 BHK','10 BHK', '11 BHK', '12 BHK', '14 BHK', '1 RK', '2 RK', '3 RK'],
    'Mapped_Value': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
})

# Replace values in df with values from mapping_df
df['No_of_BHK'] = df['No_of_BHK'].map(df_No_of_BHK.set_index('No_of_BHK')['Mapped_Value'])


# In[51]:


df.head()


# In[52]:


df['Size'].head()


# In[53]:


print(df.columns.tolist())


# In[54]:


df = df.drop(columns=['Builder_name','City_name','Locality_ID','Locality_Name','Longitude', 'Latitude','listing_domain_score'])


# In[55]:


st.write ('## Correlation Values ')
st.write(df.corr())


# In[56]:


st.write('## Correlation Heatmap')


# In[57]:


correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
plt.title("Correlation Heatmap")
st.pyplot(fig)


# In[ ]:





# In[58]:


st.write('## Ploting bar Graph of Correlation of columns with respective our Target Column(Price_USD)')


# In[59]:


correlations = df.corr()['Price_USD']
fig, ax = plt.subplots(figsize=(10, 6))
correlations.drop('Price_USD').plot(kind='bar', color='blue', ax=ax)
plt.title('Correlation with Price')
plt.xlabel('Column Name')
plt.ylabel('Correlation')
plt.xticks(rotation=90)
st.pyplot(fig)


# In[60]:


correlations.sort_values()


# In[ ]:





# ## Spliting and  training dataset

# In[61]:


st.write('# Spliting and training dataset for following models ')


# In[62]:


st.write("1. Linear Regression\n2. Decision Tree\n3. Random Forest\n4. XGBoost\n5. Support Vector Regression (SVR)\n6. k-nearest neighbors (KNN)")


# In[ ]:





# In[63]:


st.write('# Spliting and training data into 80% and 20% ratio ')


# In[64]:


X = df.drop(columns=['Price_USD'])
y = df['Price_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[65]:


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}


# In[66]:


results = []


# In[67]:


#df.dtypes


# In[68]:


#for model_name, model in models.items():
# model.fit(X_train, y_train)
#    y_pred = model.predict(X_test)

#    mae = mean_absolute_error(y_test, y_pred)
#    mse = mean_squared_error(y_test, y_pred)
#    r2 = r2_score(y_test, y_pred)

#    print(f"Model: {model_name}")
#    print(f"Mean Absolute Error: {mae}")
#    print(f"Mean Squared Error: {mse}")
#    print(f"R-squared: {r2}")
#   print()

#    results.append({
#        "Model": model_name,
#        "Mean Absolute Error": mae,
#        "Mean Squared Error": mse,
#        "R-squared": r2
#    })


# In[69]:


results = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVR', 'KNN'],
    'Mean Absolute Error': [4497985.95724051, 204511.68371606106, 136704.99382662267, 416525.36721739965, 8374648.226717024, 1229796.3902929511],
    'Mean Squared Error': [108954904351002.86, 6028801144871.48, 11560641025099.332, 18712633264889.26, 412669957211482.0, 32922449656039.785],
    'R-squared': [0.718822841527622, 0.9844416257808005, 0.9701657468939193, 0.9517087819011166, -0.06496689292513058, 0.9150378691131403]
}


# In[70]:


results_df = pd.DataFrame(results)


# In[71]:


st.write('## Evaluation Metrics of each models')


# In[72]:


results_df


# In[73]:


#results_df.dtypes


# In[74]:


model_names = results_df['Model']
metrics = ['Mean Absolute Error', 'Mean Squared Error', 'R-squared']


results_df_sorted = results_df.sort_values(by='Mean Absolute Error')
fig, ax = plt.subplots(figsize=(6, 8))
ax.barh(results_df_sorted['Model'], results_df_sorted['Mean Absolute Error'], color='skyblue')
ax.set(xlabel='Mean Absolute Error', ylabel='Model', title=f'Mean Absolute Error by Model')
ax.invert_yaxis()
st.pyplot(fig)


# In[75]:


results_df_sorted = results_df.sort_values(by='Mean Squared Error')
fig, ax = plt.subplots(figsize=(6, 8))
ax.barh(results_df_sorted['Model'], results_df_sorted['Mean Squared Error'], color='skyblue')
ax.set(xlabel='Mean Squared Error', ylabel='Model', title=f'Mean Squared Error by Model')
ax.invert_yaxis()
st.pyplot(fig)


# In[76]:


results_df_sorted = results_df.sort_values(by='R-squared')
fig, ax = plt.subplots(figsize=(6, 8))
ax.barh(results_df_sorted['Model'], results_df_sorted['R-squared'], color='skyblue')
ax.set(xlabel='R-squared', ylabel='Model', title=f'R-squared by Model')
ax.invert_yaxis()
st.pyplot(fig)


# In[77]:


metrics = ['Mean Absolute Error', 'Mean Squared Error', 'R-squared']

fig, axs = plt.subplots(3, 1, figsize=(6, 8))

for i, metric in enumerate(metrics):
    sorted_df = results_df.sort_values(metric)
    axs[i].barh(sorted_df['Model'], sorted_df[metric], color='skyblue')
    axs[i].set(xlabel=metric, ylabel='Model', title=f'{metric} by Model')
    axs[i].invert_yaxis()

plt.tight_layout()
plt.show()


# In[78]:


results_df['Index'] = range(1, len(results_df) + 1)


# In[79]:


best_model = None
best_score = float('inf')

for i in results_df['Index']:
    row = results_df.loc[i - 1]  # Subtract 1 to account for 0-based indexing
    if row['Mean Absolute Error'] < best_score:
        best_score = row['Mean Absolute Error']
        best_model = row['Model']


st.write(f"The best model is: {best_model} with MAE: {best_score}")

