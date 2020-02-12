# Mutual Fund Recommendation
John Canque

- [Data](#data)
- [EDA](#eda)
- [Model](#model)
- [Final Recommendation System](#sys)
- [Futher Steps](#steps)


## Project Goals
The goal of this project is to detect fraud in credit card transactions. With the increase use of credit cards, we will see more opportunity for fraudsters take advantage. We will use proven machine learning techniques to classify transactions for fraud. 



## Data Collection <a name='data'></a>
I used the [Kaggle Dataset] (https://www.kaggle.com/mlg-ulb/creditcardfraud) on credit card fraud. It contained 28k transactions with 492 being fraudulent. There were 31 columns; time, amount, and class were the only ones that were not masked.


## EDA <a name='eda'></a>
### Data Cleaning
The main data cleaning I did was use Robust Scaler on the amount variable and standard scaler on the time variable. I took multiple approaches in what was fed to my models with undersampling, SMOTE, and grid search being a way to tune parameters. 


### Data Exploration
As part of the data exploration, one of the more interesting things I found was fraudulent charges seemed to be just as random as the rest of the data. We also saw that the average fraudulent charge was higher than normal, but the median charge was much lower that the median for non-fraud data, indicating small transactions must have been tested first before large charges.



## Machine Learning Models <a name='model'></a>
I used logistic regression, Knears neighbors, decision tree classifier, and support vector machines to predict classifications. I also used random undersampling and SMOTE for different approaches to imbalanced data. Further I used grid search to help tune hyper parameters.
 

## Final Results <a name='sys'></a>
After the final models were ran, the best result came from K-Nearest Neighbors and grid search. This has 100% f1-score at finding the non-fraud classes and had an f1-score at finding the fraud transactions with a precision of 74% and recall of 76%.


## Further Steps <a name='steps'></a>
- Try other models: tensor flow, random forests, and other supervised learning techniques
- Try undersampling then SMOTE, then run models again
- Add functionality to take in descriptors as user input.
- Use k-fold cross validation
