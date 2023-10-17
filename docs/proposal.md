# *DATA606 - Capstone Project Proposal*
## 1. Proposal Title: Property price prediction
- **Author Name** - Balaji Manoj Jollu
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- [GitHub](https://github.com/Jollu-Balaji-Manoj)
- [LinkedIn](www.linkedin.com/in/balaji-manoj-jollu)
- Link to your PowerPoint presentation file - In progress
- Link to your YouTube video - In progress

## 2. Background
### - What is it about? 
  - Property price prediction is a significant subject in both the fields of real estate and finance. Property price prediction involves predicting the future prices of real estate properties such as homes, flats, commercial buildings, and land using data and predictive modeling approaches. This topic is critical because of the affects for many kinds of stakeholders, including homebuyers, real estate investors, property developers, and legislators.


### - Why does it matter?  

   - Property price predictions are important because they affect investment decisions, advise purchasers and sellers, influence real estate development, inform policy decisions, and serve as economic indicators. However, these predictions are loaded with risk and should be considered with other criteria for well-informed decision-making.

### - What are your research questions?

-	These questions inspire the evolution of property price prediction, allowing for better informed decision-making and the creation of precise prediction systems.
  
   1.Essential variables: we seek what variables influence property values, such as location, size, amenities, and economic factors.

   2.Model Effectiveness: They evaluate which machine learning or statistical models, such as linear regression or neural networks, are more effective in predicting prices (Linear Regression,Decision Trees and Random Forest,XGBoost or LightGBM,Support Vector,Regression (SVR),K-Nearest Neighbors (KNN)).


   3.Temporal Trends: They study if models can explain for how real estate values vary over 

   4.Considering Risk: They explore strategies to take risk and uncertainty into account when making forecasts, particularly in erratic markets.

   5.Ethical Concerns: They address possible biases in data or algorithms that may disproportionately influence specific demographic groups in property price projections.


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
  7. Rest of variables/columns except Price , Listing Category , description will be selected as features/predictors for your ML model.

## 4. Exploratory Data Analysis (EDA) 
## 5. Model Training 
## 6. Application of the Trained Models
## 7. Conclusion
 - In conclusion, our Property Price Prediction study indicates that accurate projections are critical for numerous real estate stakeholders. It emphasizes the difficulty of projecting property values, the need for data quality and preprocessing, and the importance of accounting for temporal patterns and incorporating risk. Ethical issues are acknowledged, and the study provides insights to influence real estate decision-making, with room for further research and development.
## 8. References
 1. Ja’afar, N. S., Mohamad, J., & Ismail, S. (2021). Machine learning for property price prediction and price valuation: a systematic literature review. Planning Malaysia, 19.
 2. Mohd, T., Jamil, N. S., Johari, N., Abdullah, L., & Masrom, S. (2020). An overview of real estate modelling techniques for house price prediction. In Charting a Sustainable Future of ASEAN in Business and Social Sciences: Proceedings of the 3ʳᵈ International Conference on the Future of ASEAN (ICoFA) 2019—Volume 1 (pp. 321-338). Springer Singapore.
 3. Sarip, A. G., Hafez, M. B., & Daud, M. N. (2016). Application of fuzzy regression model for real estate price prediction. Malaysian Journal of Computer Science, 29(1), 15-27.
