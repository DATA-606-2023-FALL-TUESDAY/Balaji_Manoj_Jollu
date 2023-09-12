# *DATA606 - Capstone Project Proposal*
## 1. Proposal Title: Portfolio Management using Reinforcement Learning
- **Author Name** - Balaji Manoj Jollu
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- [GitHub](https://github.com/Jollu-Balaji-Manoj)
- [LinkedIn](www.linkedin.com/in/balaji-manoj-jollu)
- Link to your PowerPoint presentation file - in progress
- Link to your YouTube video - in progress

## 2. Background
- What is it about? 
  - Reinforcement learning is a branch of artificial intelligence and machine learning that focuses on teaching intelligent agents to make decision sequences in an environment in order to maximize cumulative rewards (Predicted value). It serves as a foundation for training robots to learn from mistakes and make decisions in a manner similar to how people do.
  - Reinforcement learning algorithms utilize mathematical techniques of optimization to identify the optimum policy or value function for an agent in a specific environment. This is often accomplished through exploration and exploitation, in which the agent investigates new behaviors to learn more about the environment and then applies what it has learned to make better judgments over time.
  - Reinforcement learning has been used in a variety of fields, including robots, gaming, autonomous cars, and finance (such as portfolio management, as previously noted). It's a strong paradigm for training machines to make sequential conclusions and adapt to changing conditions, making it an important tool in the field of artificial intelligence.



- Why does it matter?  

   - Portfolio management using reinforcement learning we can predict the foliowing :

   1. Improves decision-making with data.
   2. Automates tasks for efficiency.
   3. Tailors strategies to individual goals.
   4. Manages risk effectively.
   5. Learns and adapts continuously.
   6. Scales for large portfolios.
   7. Uncovers data insights.
   8. Reduces human biases.
   9. Navigates complex markets.
   10. Drives research and innovation in finance.

- What are your research questions?
  
   1. **Data Quality**: Is our historical data reliable?
   2. **Model Robustness**: Can our model adapt to changing markets?
   3. **Risk Management**: How do we protect against significant losses?
   4. **Ethical Compliance**: Are we adhering to ethical standards?
   5. **Human Oversight**: What's the role of humans in decision-making?
   6. **Data Security**: How do we ensure data security and privacy?
   7. **Performance Metrics**: What metrics measure success?
   8. **Costs**: What are the operational costs?
   9. **Market Dynamics**: How does our model handle changing markets?
   10. **Communication**: How do we inform stakeholders?
   11. **Legal Risks**: Are there legal risks?
   12. **Long-Term Adaptability**: Is our approach adaptable to changes?

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

