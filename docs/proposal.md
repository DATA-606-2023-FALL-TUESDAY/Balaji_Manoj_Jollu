# *DATA606 - Capstone Project Proposal*
## 1. Proposal Title: Property price prediction
- **Author Name** - Balaji Manoj Jollu
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- [GitHub](https://github.com/Jollu-Balaji-Manoj)
- [LinkedIn](www.linkedin.com/in/balaji-manoj-jollu)
- Link to your PowerPoint presentation file - In progress
- Link to your YouTube video - In progress

## 2. Background
- What is it about? 
  - Reinforcement learning is a branch of artificial intelligence and machine learning that focuses on teaching intelligent agents to make decision sequences in an environment in order to maximize cumulative rewards (Predicted value). It serves as a foundation for training robots to learn from mistakes and make decisions in a manner similar to how people do.
  - Reinforcement learning algorithms utilize mathematical techniques of optimization to identify the optimum policy or value function for an agent in a specific environment. This is often accomplished through exploration and exploitation, in which the agent investigates new behaviors to learn more about the environment and then applies what it has learned to make better judgments over time.
  - Reinforcement learning has been used in a variety of fields, including robots, gaming, autonomous cars, and finance. It's a strong paradigm for training machines to make sequential conclusions and adapt to changing conditions, making it an important tool in the field of artificial intelligence.



- Why does it matter?  

   - Portfolio management using reinforcement learning we can predict the foliowing :

   1. Improves decision-making with data.
   2. Automates tasks for efficiency.
   3. Adapts techniques to meet specific objectives.
   4. Manages risk effectively.
   5. Learns and adapts continuously.
   6. Scales for large portfolios.
   7. Uncovers data insights.
   8. Reduces human biases.
   9. Navigates complex markets.
   10. Drives research and innovation in finance.

- What are your research questions?
  
   1. **Data Quality:** How can you ensure that financial data for training and assessment is reliable?

   2. **Model Selection:** What reinforcement learning methods  are best for model selection?

   3. **Reward Function:** What is the best reward function for measuring portfolio performance and risk?

   4. **Exploration vs. Exploitation:** How do you strike a balance between attempting new methods and exploiting old ones?

   5. **Risk Management:** How can you incorporate risk metrics and adapt to changing market conditions?

   6. **Overfitting:** How can you avoid overfitting while ensuring that your model adapts well?

   7. **Continuous Learning:** How does your model adapt and learn from fresh data?

   8. **Interpretability:** How can we improve the decisions made by your model in terms of their interpretability and explicability?








## 3. Data 

- **Data sources:**
  1. [List of s&p 500 Companyes](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
  2. we will download data from [Yahoo Finance](https://finance.yahoo.com/)
- **Data size:** Size of sp500_historical_data for 1 Year is 14.84 MB
- **Data shape:** 125522 Rows and 8 Columns 
- **Time period:** 1 year data
- **Data Dictionary:**
  | Column Name | Description                                                                     | Data Type  |
  |-------------|---------------------------------------------------------------------------------|------------|
  | Date        | Date                                                                            | Time Stamp |
  | Open        | Stock's initial price at the beginning of the trading day                      | Float      |
  | High        | Highest point reached during the trading day                                   | Float      |
  | Low         | Lowest point reached during the trading day                                    | Float      |
  | Close       | Stock's final price at the end of the trading day                              | Float      |
  | Adj Close   | Closing price after adjustments for all applicable splits and dividend distributions | Float   |
  | Volume      | Number of shares traded during the trading day                                 | Float      |
  | Company     | Company Name                                                                    | String     |


- Which variable/column will be your target/label and features/predictors in your ML model?
  - In this Data Except date and company all the 6 columns are used as features and features

## 4. Exploratory Data Analysis (EDA) 
## 5. Model Training 
## 6. Application of the Trained Models
## 7. Conclusion
 - In conclusion we are expecting ,Portfolio Management Using Reinforcement Learning offers interesting opportunities for optimizing investing techniques. However, it presents data, model selection, and risk difficulties that must be carefully addressed for successful implementation and long-term success.
## 8. References
 1. Grinold, R. C., & Kahn, R. N. (2000). Active portfolio management.
 2. Reilly, F. K., & Brown, K. C. (2011). Investment analysis and portfolio management. Cengage Learning.
 3. Jin, O., & El-Saawy, H. (2016). Portfolio management using reinforcement learning. Stanford University.
