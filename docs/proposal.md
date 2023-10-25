# *DATA606 - Capstone Project Proposal*
## 1. Proposal Title: Property price prediction
- **Author Name** - Balaji Manoj Jollu
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- [GitHub](https://github.com/Jollu-Balaji-Manoj)
- [LinkedIn](www.linkedin.com/in/balaji-manoj-jollu)
- Link to your PowerPoint presentation file - [Property price predictions](https://docs.google.com/presentation/d/1ARX-va2LRkcK0m8N6QgC5Mdt06Lmu0eEOjgbuWnB6FM/edit#slide=id.p1)
- Link to your YouTube video - In progress

## 2. Background
### - What is it about? 
  - Property price prediction is a crucial subject within the realms of real estate and finance. It encompasses the task of forecasting the future values of real estate assets, which can include houses, apartments, commercial properties, and land, by leveraging data and various predictive modeling techniques.


### - Why does it matter?  

   - Property price predictions are important because they affect investment decisions, advise purchasers and sellers, influence real estate development, inform policy decisions, and serve as economic indicators. However, these predictions are loaded with risk and should be considered with other criteria for well-informed decision-making.

### - What are your research questions?
  
   * Essential variables: we seek what variables influence property values, such as location, size, amenities , property type .

   * Model Effectiveness: Evaluating the effectiveness of machine learning models for predicting prices, such as Linear Regression, Decision Trees and Random Forest, XGBoost, LightGBM, Support Vector Regression (SVR), and K-Nearest Neighbors (KNN) using our dataset and problem you are working with.

## 3. Data 

  1.  This is open source data set from Kaggle - https://www.kaggle.com/code/goyaladi/property-price-ann-predictions/input
  2.  Size of data Is 180 MB 
  3. data contains 332096 rows and 32 columns
  4. we can see data is extracted from website www.makaan.com from india  recently 
  5. Data Dictionary:
     
| Column Name            | Data Type | Description                                        |
|------------------------|-----------|----------------------------------------------------|
| Property_Name          | object    | Name of Property                                   |
| Property_id            | int64     | Property id from the website                       |
| Property_type          | object    | Type of property (Apartment, Residential Plot, Villa, Independent Floor, Independent House) |
| Property_status        | object    | Status of property (Ready to move, Under Construction) |
| Price_per_unit_area    | object    | Price of unit area                                 |
| Project_URL            | object    | Website of project                                 |
| builder_id             | float64   | Builder Information                                |
| Property_building_status | object  | Status of building (ACTIVE, INACTIVE, UNVERIFIED)  |
| City_name              | object    | Name of city property located                      |
| No_of_BHK              | object    | Number of bedrooms                                |
| Locality_ID            | int64     | Zipcode of location                                |
| Locality_Name          | object    | Area of property                                   |
| Longitude              | float64   | Longitude                                          |
| Latitude               | float64   | Latitude                                           |
| Price                  | object    | Price of property                                  |
| Size                   | object    | Size of property in sq feet                        |
| Sub_urban_ID           | int64     | Neighbourhood id                                   |
| Sub_urban_name         | object    | Neighbourhood name                                 |
| description            | object    | Property description                               |
| is_furnished           | object    | Furnished or not                                   |
| listing_domain_score   | float64   | Domain score                                       |
| is_plot                | bool      | Is it plot or not                                  |
| is_RERA_registered      | bool      | Is it approved by Real Estate Regulatory Authority (RERA) |
| is_Apartment           | bool      | Is it apartment or not                             |
| is_ready_to_move       | bool      | Is it available to move or not                     |
| is_commercial_Listing  | bool      | Is it commercially listed or not                   |
| is_PentaHouse          | bool      | Is it penthouse or not                             |
| is_studio              | bool      | Is it studio or not                                |
| Listing_Category       | object    | Under which category it is listed (sell, rent)     |

  6. Price  variable/column will be your target/label in your ML model.
  7. Rest of variables/columns  are Property_type, Property_status, Price_per_unit_area, builder_id, Builder_name, Property_building_status, City_id, City_name, No_of_BHK, Locality_ID, Locality_Name, Longitude, Latitude, Size, Sub_urban_ID, Sub_urban_name, is_furnished, listing_domain_score, is_plot, is_RERA_registered, is_Apartment, is_ready_to_move, is_PentaHouse, is_studio


## 4. Exploratory Data Analysis (EDA) 

- Plot graph for  property_type with respective Price
  
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/72892f0b-e1dd-405b-8771-100d5356e0a8)

- Plot graph for  Property_status with respective Price
  
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/d5c46479-69fc-4bd2-912c-2552a4997f66)

- Plot graph for  City with respective Price

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/f97f8db6-5624-4e57-a3eb-0f741d61e8af)

- Plot graph for  Bed Room Count  with respective Price

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/b51c0dc2-208b-41bf-8804-d29e533c067c)

- Plot graph for  furniture  with respective Price

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/410d5857-1385-4ccc-b285-216f0c154878)

- Plot graph for  is Plot or not  with respective Price

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/c387c7c7-20d3-4c32-a21e-851f2def1cdf)


- Plot graph for  is Real Estate Regulatory Authority Approved or not  with respective Price

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/817351df-ee29-4db2-baab-1b03a3046074)

- Plot graph for  ready to move or not  with respective Price

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/36954794-07c1-4a59-840b-567fafc038a1)


- Plot graph for  is Apartment or not  with respective Price

  
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/79837331-2663-4864-89c8-0ed4db194ce0)


- Plot graph for  is penthouse or not  with respective Price

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/4b797a91-4e84-40cf-81a2-26ca00fb3bac)


- Plot graph for  is studio or not  with respective Price

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/8d9cebe9-a38d-497a-b5d5-5474123bf270)

-  plot showing property types in city for the dataset

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Balaji_Manoj_Jollu/assets/144192466/1dc927bd-4a9a-45fb-a677-2c632f28b3a7)



## 5. Model Training 




## 6. Application of the Trained Models
## 7. Conclusion
 - In conclusion, our Property Price Prediction study indicates that accurate projections are critical for numerous real estate stakeholders. It emphasizes the difficulty of projecting property values, the need for data quality and preprocessing, and the importance of accounting for temporal patterns and incorporating risk. Ethical issues are acknowledged, and the study provides insights to influence real estate decision-making, with room for further research and development.
## 8. References
 1. Ja’afar, N. S., Mohamad, J., & Ismail, S. (2021). Machine learning for property price prediction and price valuation: a systematic literature review. Planning Malaysia, 19.
 2. Mohd, T., Jamil, N. S., Johari, N., Abdullah, L., & Masrom, S. (2020). An overview of real estate modelling techniques for house price prediction. In Charting a Sustainable Future of ASEAN in Business and Social Sciences: Proceedings of the 3ʳᵈ International Conference on the Future of ASEAN (ICoFA) 2019—Volume 1 (pp. 321-338). Springer Singapore.
 3. Sarip, A. G., Hafez, M. B., & Daud, M. N. (2016). Application of fuzzy regression model for real estate price prediction. Malaysian Journal of Computer Science, 29(1), 15-27.
