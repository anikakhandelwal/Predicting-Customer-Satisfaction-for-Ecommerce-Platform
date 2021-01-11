# Predicting Customer Satisfaction for Ecommerce Platform
Ecommerce, also known as electronic commerce or internet commerce, refers to the buying and selling of goods or services using the internet, and the transfer of money and data to execute these transactions. It is one of the fastest growing industries in the world. Olist is the largest department store in Brazilian marketplaces. It was founded in 2015 and is based in Curitiba, Brazil. It connects small businesses from all over Brazil to channels and those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners. After a customer purchases the product from Olist Store a seller gets notified to fulfil the order. Once the customer receives the product, or the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.
# Problem Statement
E-commerce companies will benefit if more and more people place orders from their platforms or same customers order more and more products. When we place an order on an E-Commerce website our product choice and decision to buy the product gets influenced by the review and the review comments posted by customers who have bought the same product earlier. If an E-Commerce can somehow predict which products will lead better reviews and remove those products/sellers which always have worse reviews then it will be helpful in increasing the overall customer satisfaction.
# Mapping to a Machine Learning Problem
The above problem can be mapped to a two-class classification problem. The objective of the problem will be to classify a review into a positive review or a negative review. The performance of the models can be judged on the basis of 
1. Confusion matrix 
2. F1 Score
# Dataset
We have a dataset which has information of 100k orders from 2016 to 2018 placed at Olist Store. The dataset has many important details of the order like Product Type, Price, Seller Details, Delivery Date, Review Score, Review Comments etc. The dataset is publicly available on Kaggle. It is divided into multiple datasets.
Schema of the Dataset
![alt text](https://i.imgur.com/HRhd2Y0.png)
We will use this dataset to train a Machine Learning Model which can effectively classify a review as a positive or a negative review. 
# References
    1. https://www.kaggle.com/olistbr/brazilian-ecommerce
    2. https://www.kaggle.com/thiagopanini/e-commerce-sentiment-analysis-eda-viz-nlp
    3. https://www.kaggle.com/souravbarik/customer-satisfaction-using-olist-dataset
    4. https://www.kaggle.com/duygut/brazilian-e-commerce-data-analysis 
    5. https://www.kaggle.com/andresionek/understanding-the-olist-ecommerce-dataset
