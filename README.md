# Bank churner prediction using pycaret

## DataSet Review

1. Task Details: To predict the customers who are churned or not churned using method of classification based on the customer_churn(existing means 0/ attrited means 1). 

    ***The meaning of attrition customer is the result of clients and customers terminating their relationship with a brand.

2. Target: customer_churn: if Existing (not churn) means 0 and Attritied (churn) means 1.

3. Data Contain: The dataset of BankChurner consists of 23 columns and 10127 rows in total.
 
4. There are total 16.1% of churn customer and 83.9% of not churn customer,  which means the dataset is imbalance. The probability threshold adjustment shall apply to solve the imbalance dataset.

## Customer Churn Definition
1. Customer churn refer to the percentage of customers who decided to stop doing the business or using service with your company. 

2. To retain an existing customer is much easier than acquire new customers as the existing customer is ease to be convince with least time and cost. 

3.  The prediction for customer churn is importance to prevent rate of churn increase.

## Bank Customer Churner Analysis Process
   ![image](https://user-images.githubusercontent.com/59326036/141981227-9e665c69-fcee-433a-a365-ea6cb338ebaf.png)
   

## Categories of Features In Dataset
   ![image](https://user-images.githubusercontent.com/59326036/141981803-27450128-780b-4d75-a3b7-7241dd21df78.png)
   
## Target
To predict the bank customers in a bank whether to …… 
   ![image](https://user-images.githubusercontent.com/59326036/141981964-6bc3b2ef-3cdc-47f0-b697-66305ffae08d.png)

## First, setup the model with pre-processing

```
from pycaret.classification import *

# try with z-score normalization
# z-score normalization is similar to StandardScaler in sklearn
model = setup(
    data=df,
    target='customer_churn',
    categorical_features=cat_vars, 
    ignore_features=['clientnum'],

    normalize=True, #pre-processing
    normalize_method='zscore',  #pre-processing
)

```

```
#compare the model before create model
compare_models(sort='F1')
```
![image](https://user-images.githubusercontent.com/59326036/141990878-398f8319-12d7-45a4-9ac5-9801213f9c13.png)


```
#create best model from above
best_model = create_model('lightgbm')
```
![image](https://user-images.githubusercontent.com/59326036/141990721-881d6255-7a2f-49db-9376-2dd5d6d90e05.png)


## Tune model to get better F1 score
Tune model to get best F1, the mean of lr_tuned is slighly higher than best_model. Thus, decided to use the lr_tuned.
```
params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
          'max_iter': [10, 100, 1000], #max number of iteration
          }

lr_tuned = tune_model(best_model,
           n_iter=50,
           optimize='F1',
           custom_grid=params
           )        
```
![image](https://user-images.githubusercontent.com/59326036/141990599-15929c03-cabf-460b-b1b3-43b2e69dc45a.png)

#### Lightgbm F1 score between tune and before tune 
![image](https://user-images.githubusercontent.com/59326036/141990509-a23eeb6f-870f-4702-a97a-8bb1bdc5b082.png)

#### Features explanation
Based on the graph at below, we can see the total transaction amount, total amount change for Q4 over Q1 & total transaction count having higher impact on customer intend to churn. The bank shall start to implement some solution such as online survey or analysis the data to understand better regarding why customer likely to churn. This reason could be loan interest rate not much attractive or credit card restriction too tight for customer.

The customer age also huge impact on customer intend to churn, mostly the age group between 40 to 50 will likely to churn. The bank should exam the root cause of churn for this group of customer. Maybe they suffer in family commitment, so they intend to change other bank.

The promotion shall implement including to attract customer to stay with bank:

1. Up to one handred cashback with at least one trasaction.
2. An Apple watch to customer who spend with a mininum of three hundred.
3. Two hundred e-voucher with a mininum spend of $300 and etc.

The bar chart at below showing the trend of churn on group of age. Most likely age group between 40 to 50 intend to churn.
```
#As observation, the age between 40 to 50 are likely to churn.
def drawAge():
    age = df.groupby(df['customer_age'])['customer_age'].count().reset_index(name = 'Total')
    age01 = age.values.tolist()

    #print(age01)

    dfplot = pd.DataFrame(age01, columns = ['customer_age', 'Total'])
    #print(dfplot)

    #Display the results in histogram based on dfplot dataframe.
    dfplotHis = plt.figure(figsize=(15,6))
    dfplotAxes = dfplotHis.add_subplot(1,1,1)
    dfplotAxes.bar(dfplot['customer_age'],dfplot['Total'], color= '#00008B', alpha=0.7)
    dfplotAxes.grid(axis='y', alpha=0.75)
    dfplotAxes.set_title('Bar Chart for Count of each age group', fontsize=15)
    dfplotAxes.set_xlabel('Age group', fontsize=15)
    dfplotAxes.set_ylabel('Count of age ', fontsize=15)
    plt.xticks(rotation=70) #rotate the word for x axis.

    plt.show()  
drawAge()
```
![image](https://user-images.githubusercontent.com/59326036/141991599-65a4e539-25fe-4a88-a8e6-147bd1bce97c.png)

```
# The 10 most important feature that sorted by highest scores for prediction
# plot the model with parameter = feature
plot_model(lr_tuned, plot='feature')
```
![image](https://user-images.githubusercontent.com/59326036/141991802-7f8667c0-7cf8-4693-b010-9edf443951ba.png)

```
#plot ROC-AUC chart.
plot_model(lr_tuned)

```
![image](https://user-images.githubusercontent.com/59326036/141992410-f10ade02-bdb2-4545-8aa5-4bc534e4e0fe.png)

#### Explaination of confusion matrix. 
1. True positive (TP): Means the cases in which predicted yes and customer is churn. How often the predicted is yes.

2. True Negative (TN): Means the cases in which predicted no and customer is not churn. How often the predicted is No.

3. False negative (FN): Means the predicted no but actually the customer is churn.

4. False positive (FP): Means the predicted yes but actually the customer is not churn.

#### Recall Or Precision?
1. Precision is refer to the correctly positive cases out of all predicted as positive, which is how precise of actual positive when it predicts as positive. For instance, out of 10 customer predicted as churn by the model, how many customer are correct churn.

3. Recall is refer to the number of true positives cases out of all positives in dataset, which is how often predicted correctly as positive. For instance, out of 10 churn customer in the data, how many customer classified correctly as churn customer.

3. As observation at above, we shall using recall for confusion matrix measurement. This reason is we wish to find out how many customers are correctly predicted as churn from 10127 rows in this dataset.

```
#plot confusion matrix chart.
evaluate_model(lr_tuned)
```
![image](https://user-images.githubusercontent.com/59326036/141993301-26a3459d-eeaf-414e-890d-8e1326137592.png)

## Save the model 
```
# saving the model will save it to the runtime environment
save_model(lr_tuned, model_name='bank_churner_train_01')
```

## Prediction for the model
```
#load saved trained model
best_model = load_model('bank_churner_train_01')
```
#### Try retrieved sample of data for customer and test the prediction score.
```
#check the columns and use for sample of user request.
print(df.columns)
df.iloc[0] 
```
![image](https://user-images.githubusercontent.com/59326036/141994640-5e71bbe6-4f13-4b15-b2c6-4ab151c0be0e.png)

#### Start to perform prediction.
```
predict_model(best_model, user_request)
```
![image](https://user-images.githubusercontent.com/59326036/141995112-6a837277-ac16-4c31-99fd-7e380d3c5179.png)
