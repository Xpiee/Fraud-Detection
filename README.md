# Table of Contents

[1IEEE-CIS Fraud Detection (Kaggle Competition) 3](#_Toc26304249)

[1.1Problem Definition 3](#_Toc26304250)

[1.2Objective 3](#_Toc26304251)

[1.3Data Description 3](#_Toc26304252)

[1.3.1Transaction Table 3](#_Toc26304253)

[1.3.2Identity Table 4](#_Toc26304254)

[1.4Methodology 5](#_Toc26304255)

[1.4.1Data Understanding 5](#_Toc26304256)

[1.4.2Challenges and its Solutions 9](#_Toc26304257)

[1.4.3Evaluations 14](#_Toc26304258)

[2Conclusion and Future Work 17](#_Toc26304259)

[2.1Conclusion 17](#_Toc26304260)

[2.2Future Work 17](#_Toc26304261)

# 1IEEE-CIS Fraud Detection (Kaggle Competition)

## 1.1Problem Definition

Fraud is a billion-dollar business and it is increasing every year. According to CFCA 2017 Survey, 1.4 billion dollars were estimated to be lost in just payment fraud in the year 2017.The PwC global economic crime survey of 2018 found that half (49 percent) of the 7,200 companies have experienced fraud of some kind.This is an increase from the PwC 2016 study in which slightly more than a third of organizations surveyed (36%) had experienced economic crime. These frauds are prevented by the researches from IEEE Computational Intelligence Society (IEEE-CIS).

The fraud detection system is actually saving lot of consumers millions of dollars per year. Researches from IEEE Computational Intelligence Society (IEEE-CIS) want to improve this figure, while also improving the customer experience. With higher accuracy fraud detection, we can get on with our chips without the hassle.

## 1.2Objective

The objective of our project is to train a model on a challenging large-scale dataset that can predict the probability that an online transaction is fraudulent or not. The dataset is provided from _Vesta Corporation_ that comes from Vesta real-world e-commerce transactions. Based on the information provided, we understand that this is a Binary classification problem i.e., we need to just predict if the transaction is fraudulent or not fraudulent.

## 1.3Data Description

The dataset is provided by Vesta Corporation, namely Transaction Table, and Identity Table having a primary key as 'TransactionID'. The data sources are given as 'sample\_submission.csv' [507k x 2], 'test\_identity.csv' [142k x 41], 'test\_transaction.csv' [507k x 393], 'train\_identity.csv' [144k x 41] and 'train\_transaction.csv' [591k x 394].

### 1.3.1Transaction Table

The Transaction dataset provides the list of transactions like money transfer, ticket booking, purchase, etc. It has following attributes –

1. TransactionDT: It is the timedelta which is based on the reference datetime and is not an actual timestamp.
2. TransactionAMT: It is the payment transacted in USD.
3. ProductCD: It is the code of the product for which the transaction has been made.
4. card1 – card6: It is the information of the card for which the transaction has been made.
5. addr1 and addr2: It is the addresses of the purchaser.
6. dist1 and dist2: It is the distance between the location of two transactions.
7. P\_emaildomain and R\_emaildomain: It is the email domain of the purchaser and the recipient.
8. C1 – C14: It is the count of the addresses associated with the card.
9. D1 – D15: It is the time-delta specifying the days between the previous transaction.
10. M1 – M9: It is the match such as name on card and address, etc.
11. V1 – V339: It is the information provided by the Vesta specifying the entity relations.

![](RackMultipart20220817-1-18b5rb_html_c0a6431937e153e.png)

Fig 1: Training transaction dataset

The categorical features in the transaction table are ProductCD, P\_emaildomain, R\_emaildomain, card1 - card6, addr1, addr2, M1 - M9.

### 1.3.2Identity Table

The Identity Table provides us the information about the network connections (IP, ISP, Proxy, etc.). It has following attributes –

1. DeviceInfo: It is information of the device from which transaction has been made.
2. Devicetype: It is the type of device (eg- Mobile phone, card payment machine, etc) from which payment has been made.
3. id1 – id38: It is the entity features collected by the Vesta fraud protection system and digital security partners.

![](RackMultipart20220817-1-18b5rb_html_2149ba4b25c76934.png)

Fig 2: Training identity dataset

The categorical features in the identity table are DeviceType, DeviceInfo and id\_12 to id\_38

## 1.4Methodology

We have performed various steps to train a machine learning model that can predict if the transaction is fraudulent or not. We have divided these steps into 4 major categories. These categories are mentioned below:

1. Data Understanding
2. Challenges and their Solutions
3. Training the Machine Learning Model
4. Evaluation

### 1.4.1Data Understanding

To understand the dataset properly, it very important to explore, observe and draw insights from the data. This would help us to identify what are the issues in the data and how we can prepare the data to apply the modelling algorithm.

In order to do that, firstly we have checked the statistics of all the files provided to us. This helped us in getting brief information of the data.

1. From training identity data, we found that for id\_02 has minimum value as 1, maximum value of 999595 and mean is of 174716, that means outliers can be present. For id\_03 to id\_06, most of the values lies in the first quarter only and it has a greater number of negative values. These things we have analyzed in detail in later section.

![](RackMultipart20220817-1-18b5rb_html_60b53564ff4a0182.png)

Fig 3: Description of Training Identity table

1. For test identity table, similar anomalies as training identity dataset were seen.
2. For test transaction dataset, we observed that the attribute TransactionDT need to be split in hours and days for better understanding of data. TransactionAmt may have outliers because of the huge difference between the maximum and mean value.
3. For test transaction dataset, we observed similar kind of anomalies as the test transaction table.

Secondly, we identified the missing values in the training dataset. We checked the percentage of missing values in the dataset and found that 41% of the data is missing in the transaction table and 35.5% of the data is missing in the Identity table.

![](RackMultipart20220817-1-18b5rb_html_7874228acb2a5e60.png) ![](RackMultipart20220817-1-18b5rb_html_3639f7b6c640ad4f.png)

Fig 4: Transaction table missing values Fig 5: Identity table missing values

We also found the missing values in the percentage of the total values. Through this we identified that the features dist2, D7, D13 and many more have more than 85% of the missing value. These features won't be helpful to us for the training the model.

![](RackMultipart20220817-1-18b5rb_html_aa7d18ab46e19bb1.png)

Fig 6: Line plot of missing values in training identity table

The transaction table attributes can be roughly separated into groups by their missing value distribution (V1,V11) (V12,V34) (V35,V51) (V52,V74) (V75,V94) (V95,V137) (V138,V166) (V167,V278) (V279,V321) (V322,V339)

Then we identified the correlation between the features in both the training dataset. We found that some features like V4 and V5, V10 and V11, etc are highly correlated.

![](RackMultipart20220817-1-18b5rb_html_b5deeb0237028885.png)

Fig 7: V attributes correlation in the training transaction dataset

We used Dendogram to identitfy the correlation in the identity table. We found that id\_17 and id\_19, id\_30 and id\_32, id\_03 and id\_04, etc are correlated in the table. Using one feature for each scenario will give same result during training.

![](RackMultipart20220817-1-18b5rb_html_47283ea4e5912b83.png)

Fig 8: Correlation between features of identity table

We also checked for the imbalance data for the training dataset. We found that that 3.5% of the data is fraudulent while 96.5% of the data is non-fraudulent. The fraudulent transactions are too small compared to the real ones.

![](RackMultipart20220817-1-18b5rb_html_2adb8e312fe5bbdc.png)

Fig 9: Imbalance dataset

We also checked the skewness of the training dataset and found that most of the data in the training identity dataset is rightly skewed.

![](RackMultipart20220817-1-18b5rb_html_37538d2d0e342037.png) ![](RackMultipart20220817-1-18b5rb_html_63af5f2b0d7f50bb.png)

Fig 10: Skewness for each attribute Fig 11: Skewness of training identity table

For each feature, shown in fig 10, when skewness is less than -1 or greater than +1, the distribution is highly skewed like id\_09, id\_22. If it is between −1 and −½ or between +½ and +1, then the distribution is moderately skewed like id\_32. If it is between −½ and +½, then distribution is approximately symmetric like id\_17, id\_19.

We also checked for the distribution of the transaction amount to identify how much amount is getting transacted in fraudulent and non-fraudulent.

![](RackMultipart20220817-1-18b5rb_html_d2130d3bd98604b.png)

Fig 10: Comparison of fraud and non-fraud data in training transaction dataset

We found that the amount of non-fraudulent transactions is higher than the fraudulent which is which is not possible in the practical scenario.

We also looked for the V-attributes distribution. They are the largest set in the table and are vesta engineered features.

![](RackMultipart20220817-1-18b5rb_html_124e09e7db9256be.png) ![](RackMultipart20220817-1-18b5rb_html_f28df816b98ec14f.png) ![](RackMultipart20220817-1-18b5rb_html_bc2491f9d5eb61de.png) ![](RackMultipart20220817-1-18b5rb_html_d8131d4c89567457.png)

Fig 11: Distribution of V-attributes

We identified that the distribution of few features is almost similar like V2 and V5, V7, V8 and V9. These features were less helpful in determining whether the transaction is fraud or not

### 1.4.2Challenges and its Solutions

The first and foremost challenge we faced was the memory consumption. The dataset provided was extremely large so while applying any operation on it. It was consuming time and memory.

In order to reduce memory consumption of the whole process we did two major things

1. Merged the data of the training transaction and training identity dataset.
2. Reduced the datatypes of all numerical features whether it is 'int64', 'int32', 'float16' or 'float32' based on the feature values.

![](RackMultipart20220817-1-18b5rb_html_38559154dc089aa2.png)

Fig 12: Reduction of memory for training and test dataset

As mentioned earlier, 'TransactionDT' need to be split into hours and days for better understanding. So, at first, we compared its distribution in training and test dataset. From this, we identified that the test data is ahead of training data and the initial timing of training data is high and last timings of test data is high.

![](RackMultipart20220817-1-18b5rb_html_355f1bb1c44c14f0.png)

Fig 13: Distribution of 'TransactionDT'

Then, we did timeseries split in hours and days since 'TransactionDT' is given in seconds. We checked further that which day is having more fraud than the other in both test and training dataset.

![](RackMultipart20220817-1-18b5rb_html_c087508d104c6d1d.png) ![](RackMultipart20220817-1-18b5rb_html_46dba016ef824991.png)Fig14: Distribution of transactions on Fig 15: Distribution on the basis of fraud

the basis of day and non-fraud transactions in a day

From the Fig 14, it was identified that the Day 0 has the greater number of transactions than the other days. But, on the basis of fraud there was equal percentage of fraudulent and non-fraudulent transactions. So that's why we plotted bar graph on the basis of hours which showed that in the hours 4 - 9 more fraudulent transactions happen while in the hours 13-17 less fraudulent transactions happen.

![](RackMultipart20220817-1-18b5rb_html_befd847a2d71fe1e.png)

Fig 16: Distribution on the basis of fraud

and non-fraud transactions in an hour

For the Transaction Amount, we first plotted the scatterplot on the basis of TransactionDT which was helpful in order to find the outliers.

![](RackMultipart20220817-1-18b5rb_html_97a601e327fabd31.png)

Fig 17: Distribution of TransactionAmt

on the basis of TransactionDT

From the plot, we found that there are values above 6000 as well. So, we decided to set the threshold value and above that we considered it as outliers. We took 10000 as the threshold value and above that only 2 values were present and both of them were duplicates too, so we deleted them.

In the next step, we analyzed the ProductCD on the training dataset. We plotted the bar graph in order to check which product code is having a greater number of transactions and more of fraudulent transactions. We found that the product code 'C' has very high chances of transaction being fraud. On the other hand, with 'W' product code chances are low and with product codes 'H', 'R', 'S' chances are moderate. While most of the transactions were happened for the product code 'W'.

![](RackMultipart20220817-1-18b5rb_html_d323f6c9a7d308e4.png)

Fig 18: Distribution of the ProductCD on training dataset

Similar type of analyses we did for Card 1 to Card 6. For Card1 and Card2 nothing we could deduce from it. For Card3 only two values were present in majority and greater than the value 150 fraudulent transactions increased. For Card4 we were able to determine that 'discover' card is having a smaller number of fraudulent transactions and for others it is relatively equal. Most of transactions involved 'visa' and 'mastercard'. For Card5, the distribution was varying too much so nothing we were able to infer. For Card6, credit card holders had greater number of fraudulent transactions while for 'charge card' and 'debit or credit', there were no transactions only, so we classified it as 'others'.

![](RackMultipart20220817-1-18b5rb_html_63150bd562fad5ee.png) ![](RackMultipart20220817-1-18b5rb_html_75d89e8e72beea59.png)

Fig19: Distribution of Card4 on the Fig 20: Distribution of Card6 on the basis

basis of fraud and non-fraud fraud and non-fraud

For attribute'Addr1', the values were varying too much, so we were not able to infer anything from that. For 'Addr2', we found that the value 87 was having a greater number of transactions in the training as well as in test dataset.

For attribute 'Dist1' and 'Dist2', we found outliers in the training dataset by plotting the scatterplot on the basis of transactions. The values above 6000 in 'Dist1' and values above 8000 in 'Dist 2' were removed. But 'Dist2' was also having 90% of the null values, so this feature is of no use for modeling.

For 'P\_emaildomain' and 'R\_emaildomain', we split them on the basis of domain and extension and then plotted the bar graph. We found that most of the fraudulent transactions happened through the gmail domain in both the attributes for the training dataset.

![](RackMultipart20220817-1-18b5rb_html_213e41f477af4838.png) ![](RackMultipart20220817-1-18b5rb_html_f50e9aa55b246910.png)

Fig 21: Distribution of p\_emaildomain Fig 22: Distribution of r\_emaildomain

on the basis of fraud and non-fraud on the basis fraud and non-fraud

While analyzing C attributes, outliers were present. So, we removed it from the dataset. For C1 and C2 values above 2000 were removed. C3 had 75% of the value as null, so attribute was removed. In C4 and C5, we were not able to infer anything. But we found distribution of C4 and C6 almost similar when plotted. So, we plotted the heatmap for the finding the correlation between C attributes and found that C1 and C2, C1 and C6, C1 and C14, C6 and C14 are highly correlated. So, while training the model we considered only one of the attributes.

We also plotted the heatmap for the D attributes and found that D4 and D12, D6 and D12, D5 and D7 are highly correlated. So, only one attribute was taken into consideration for modeling. But, D7 was having 90% of the null values so it removed completely. We were not able to make any deductions from the M attributes. Similarly, for the 'DeviceInfo', most of the transactions happened through 'Windows' and 'iOS Device'. For 'DeviceType', same number of fraudulent and non-fraudulent transactions for both 'desktop' and 'mobile'

Based on the above analysis, following are the attributes that we did not consider in our algorithms.

| **Characteristics of Attributes** | **Attribute Name** |
| --- | --- |
| Almost one value (90%) | 'dist2', 'C3', 'D7', 'V98', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123','V124', 'V125', 'V129', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V281', 'V284', 'V286', 'V290', 'V293', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V305', 'V309', 'V311', 'V316', 'V318', 'V319', 'V320', 'V321', 'id\_07', 'id\_08', 'id\_18', 'id\_21', 'id\_22', 'id\_23', 'id\_24', 'id\_25', 'id\_26', 'id\_27' |
| Many null value (90%) | 'dist2', 'D7', 'id\_07', 'id\_08', 'id\_18', 'id\_21', 'id\_22', 'id\_23', 'id\_24', 'id\_25', 'id\_26', 'id\_27' |
| Dropped Attributes | 'id\_01','id\_02','id\_03','id\_04','id\_05','id\_06','id\_10','id\_11', 'id\_13', 'id\_14', 'id\_17', 'id\_19', 'id\_32', 'P\_emaildomain','R\_emaildomain','id\_07','id\_08', 'id\_09', 'id\_12', 'id\_15', 'id\_16', 'id\_18', 'id\_20', 'id\_21', 'id\_22', 'id\_23', 'id\_24', 'id\_25', 'id\_26', 'id\_28', 'id\_29', 'id\_30', 'id\_31', 'id\_33', 'id\_34', 'id\_35', 'id\_36', 'id\_37', 'id\_38', 'DeviceInfo' |

![](RackMultipart20220817-1-18b5rb_html_a8f1c7d0db389296.png)

Fig. 23: Showing how PCA is separating dataset in a linear format

### 1.4.3Evaluations

In this analysis, we have trained our model on Logistic Regression, SVM, Light GBM, RandomForest and XGBoost classifiers. For training our models on the dataset, we applied Label Encoding to the following attributes: 'card4', 'hour', 'card6', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'ProductCD', 'DeviceType', 'DeviceName', 'Check' ('Check' is an engineered feature from the column 'hour'). Along with it, we used Standard Scaler (from sklearn) to Standardize features by removing the mean and scaling to unit variance.

Below are the results of the trained models on the test set. For more details you can follow the attached python file with all the code along with comments with each segment.

1. Logistic Regression:

For Logistic Regressions, we were able to achieve AUC score of 82.20% on the test dataset.

![](RackMultipart20220817-1-18b5rb_html_d470e2e2fcf74f5e.png)

Confusion Matrix:

| Actual | Predicted |
| --- | --- |
| 0 | 1 |
| 0 | 157417 | 1985 |
| 1 | 4373 | 1340 |

1. Support Vector Machine (SVM):

For SVM, we got a very low AUC score of 59.98% on the test dataset.

![](RackMultipart20220817-1-18b5rb_html_5672d3d1593ee472.png)

Confusion Matrix:

| Actual | Predicted |
| --- | --- |
| 0 | 1 |
| 0 | 158038 | 1364 |
| 1 | 4523 | 1190 |

1. Light GBM:

For LGBM model, we were able to achieve an AUC score of 92.10% on our validation dataset. To train this model we used the following hyperparameters [5]:

![](RackMultipart20220817-1-18b5rb_html_8d6c41d25a5da4a9.png)

![](RackMultipart20220817-1-18b5rb_html_b474e89e9ad7f459.png)

1. Random Forest:

For Random Forest, with default parameters, we were able to achieve an AUC score of 65.27%.

Confusion Matrix:

| Actual | Predicted |
| --- | --- |
| 0 | 1 |
| 0 | 159020 | 382 |
| 1 | 3954 | 1759 |

1. XGBoost:

For XGBoost, we were able to train the model with the validation-error of 2.9%. To train the model, we used the following hyperparameters:

![](RackMultipart20220817-1-18b5rb_html_fd3096e0ce9ac5f.png)

Using the above hyperparameters, we achieved the following results:

![](RackMultipart20220817-1-18b5rb_html_b53a44d8c6832399.png)

# 2Conclusion and Future Work

## 2.1Conclusion

Based on our analysis, we can say that LGBM was unaffected to the problem of imbalanced dataset and was able to accurately classify the transactions into fraud and non-fraud transactions with 92% accuracy. By understanding the time series nature of the whole dataset, we were able to extract some important features from the dataset by dividing the transaction time into days and hours and then by observing the pattern of transactions (fraud and non-fraud transactions) with the 'hour' at which those transactions where completed. Using this pattern, we were able to extract a very useful insight that transactions completed between hour 3 to 10 had more percentage of fraudulent transactions than the legit transactions. We believe that using this information in our model helped us achieve better accuracy.

Also, we initially believed that the XGBoost would be able to handle this problem better than LGBM; however, after training our model using XGBoost, we think that the model might be overfitting as it saturates very early and the validation error does not improve. In our view, LGBM was able to better classify the dataset without being affected by data imbalance and did not overfit.

## 2.2Future Work

It is evident that LGBM's accuracy can be increased using more advanced techniques. For future work, we think that following techniques can be explored a bit more for improving the overall accuracy of the model.

1. Principle Component Analysis (PCA): Using Principle Component Analysis to reduce the dimensionality of the dataset. This would definitely improve the overall accuracy of the trained model.
2. Vesta Features: Most of the Vesta features where having more than 90 percent of missing values and other were correlated with each other. We would like to study the Vesta features in much more details so that important features can be extracted from these features. Also, PCA can be implemented on the Vesta features to reduce these features into attributes with more information.
3. XGBoost: XGBoost is computationally expensive and it requires GPU to train the XGBoost model. So, we would like to refine our dataset using the above-mentioned techniques and then would like to train the XGBoost model using GPU to check the performance of this model on fraud-transaction classification problem. We still believe that XGBoost would outperform the LGBM model if trained on more refined features given all the resources required to train a XGBoost model.
4. Hyperparameters Tuning: In future, we would like to further work on the trained model by fine tuning the hyperparameters used from LGBM and XGBoost using grid search with cross-validation to optimize the trained model performance.

References:

1. Fraud complete EDA. link: [https://www.kaggle.com/jesucristo/fraud-complete-eda#Time-vs-fe](https://www.kaggle.com/jesucristo/fraud-complete-eda#Time-vs-fe)
2. Scikit Learn – LogisticRegressionCV. link: [https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.LogisticRegressionCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
3. Logistic Regression Model Tuning with scikit-learn. link: [https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5](https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5)
4. XGBoost Documentation. link: [https://xgboost.readthedocs.io/en/latest/](https://xgboost.readthedocs.io/en/latest/)
5. KaustuvDash - IEEE-Fraud-Detection. Link: [https://github.com/KaustuvDash/IEEE-Fraud-Detection](https://github.com/KaustuvDash/IEEE-Fraud-Detection)
6. Extensive EDA and Modeling XGB Hyperopt. Link: [https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt](https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt)
7. EDA and models. Link: [https://www.kaggle.com/artgor/eda-and-models](https://www.kaggle.com/artgor/eda-and-models)
