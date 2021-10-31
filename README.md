# Material back orders
Material back orders are a common problem in supply chain management system. It happens when a demanded product has gone out of stock and the retailer fails to fulfill the order. This article is based on a machine learning case study which predicts the probability of a given product to go on back order using ensemble ML algorithms. The ML problem was appeared in the Kaggle Challenge "Predict Product Backorders".
# Machine Learning Problem:
Machine learning models that are learned from past inventory data can be used to identify the materials at risk of back order before the event occurs. The model identifies whether a product will go on back order or not. So this problem can be mapped into a binary classification problem which classifies the products into
● products that tend to go on back order as positive class('went_on_backorder', labelled 1)
● Products that are less likely to go on back order as negative class(labelled 0)

# Evaluation metrics:
AUC : ROC Curve is plotted between false positive rates and true positive rates obtained by the model predictions at various thresholds. Area Under the ROC curve can be interpreted as the probability that a given classifier ranks a random positive example above a random negative example. An ideal classifier gives an AUC value of 1 whereas a random classifier gives an AUC of 0.5. AUC less than 0.5 denotes a classifier worse than random classifier. Here AUC is chosen as the primary metric

2.Precision,Recall and F1 Score:
 ● Precision indicates how confident the model in its prediction and
● Recall indicates how well the model is in predicting all positive points correctly.
Here we are interested in detecting as much as points that tend to go on back order,ie., the positive class. Recalling all the positive points correctly is important here because the cost of miss classification is higher for not detecting a back order(low recall) than labeling a product that won't go back order as a product that would go back order ( low precision)( However,this could vary with the domain). But we can't compromise too much on precision. So we choose F1 score ,which is the harmonic mean of precision and recall, keeping an eye on recall.

# The Data set
The data set was appeared in Kaggle's competition ' Predict Product Backorders' , and is now available in this github profile. It contains ~1.9 million observations ,22 features and the target variable. The features are:
● sku: unique product id
● national_inv- Current inventory level of components.
● lead_time -Transit time between placing the order at manufacturer and
receiving at retailer.
● i n_transit_qty - Quantity in transit
● forecast_x_month - Forecast sales for the net 3, 6, 9 months
● sales_x_month - Sales quantity for the prior 1, 3, 6, 9 months
● min_bank - Minimum recommended amount in stock ( safety stock)
● potential_issue - Indicator variable noting potential issue with item
● pieces_past_due - Parts overdue from source
● perf_x_months_avg - Source performance in the last 6 and 12 months
● local_bo_qty - Amount of stock orders overdue
● X17-X22 - General Risk Flags
● went_on_back_order - Whether the product went on back order, Target
Variable

# Challange: Imbalanced data
In the data set , more than 99% of the observations belong to class 0 .ie., majority of products did not go on back order. Only a small percentage of products did go on back order. Clearly, the data set is highly imbalanced with percentage of minority class less than 1.

# Model:
custom stacking model was built with an auc of 96.39 and recall of 0.89 , and deployed using streamlit. The App is running on https://share.streamlit.io/nimishamohanv/material-backorder-prediction/main/app.py.
