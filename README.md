# Capstone_Two

![image](https://user-images.githubusercontent.com/82134417/133855823-29ace47c-4935-4ad1-884a-e3aa668b934f.png)  

*Airbnb is one of the innovative phenomena in the sharing economy. It enjoys around 5% of the tourism accommodation revenue. However, setting a price for their listings has been a challenging task for Airbnb hosts. Because unlike in hotel industry, prices for their listings have been handled by hosts themselves, which sometimes lead to inefficient pricing followed by financial losses. This project, attempts to produce a pricing model to help new hosts in the New York area to determine a price for their listing based on several features. That is, this considers only the features that are inherent to the listing. For instance, the location of listing, type of housing, rating, available amenities, reviews about the listing, reviews etc. Also, this will help the hosts to identify which aspects or features should be given more priority over others to offer value for the price they are going to charge.*

## 1. Data  
Data was obtained from an online data site called insideairbnb, which sources publicly available Airbnb listings’ data using APIs. Data consisted of 36,000 listings (36,000 rows) from 5 different Boroughs of New York such as Manhattan, Brooklyn, Queens, Bronx and Staten Island. This data covers early periods of April 2021.

## 2. Data Wrangling and Exploratory Analysis
It was determined that missing data was missing at random only and missingness < 10% were imputed with suitable values.Following features were retained for further analysis.

[Data cleaning procedures](https://github.com/BS12345678/Capstone_Two/blob/main/01_ABNB_DataWrangling.ipynb)  

![image](https://user-images.githubusercontent.com/82134417/133856343-9b4d1b74-4b9a-424f-a4bb-d39accac5360.png)

Following are some plots showing the distribution of features and some associations among them.

![image](https://user-images.githubusercontent.com/82134417/133856452-6abcbefc-9f01-4c38-b25f-501f87ada8ba.png) | ![image](https://user-images.githubusercontent.com/82134417/133856479-8af08271-8eeb-4805-870f-4258394ee6c0.png)

![image](https://user-images.githubusercontent.com/82134417/133856672-d62d610e-2981-4d9b-866d-86eaac14ce10.png)  

![image](https://user-images.githubusercontent.com/82134417/133856724-53552bd7-6be5-4a0a-bf4e-25787a48b67c.png)  

![image](https://user-images.githubusercontent.com/82134417/133856781-16bdb593-689a-4fb2-9de2-177a42030162.png)  

![image](https://user-images.githubusercontent.com/82134417/133856841-b793d9d5-b3c5-4249-a509-221e8f0738d2.png)  

![image](https://user-images.githubusercontent.com/82134417/133856881-c93aa703-c525-4703-908f-9b8b7db1a696.png)  

With target feature price, 35 listings had 0 prices, so it had to be removed. Prices after 99th percentile were also removed as it consisted outlier values like $2,000. Now the price ranged between $20 and $946. However, still the price showed an extremely skewed distribution with 75% of samples ranged between $20 - $156 and rest of the percentile ranged between $156 and $946. In order to rectify this, price was log-transformed.

![image](https://user-images.githubusercontent.com/82134417/133856950-a8ce1ed9-8139-43a6-bab6-b7b25b19cf2d.png)  

## 3. Modeling
[Modeling Notebook](https://github.com/BS12345678/Capstone_Two/blob/main/04_ABNB_Modeling.ipynb)

Before modeling, preprocessing steps like splitting train and test data (test-size was 0.25) and standardization of numeric features were done. Different models with incremental complexity were used in predicting log prices. Mean Absolute Error (MAE) was used as the performance metric for each model. In this project, MAE measures the average difference between the actual price and the predicted price using the model. I chose MAE because it is relatively easy in terms of understanding from the business point of view. Below table illustrates how MAE changed with different algorithms.

![image](https://user-images.githubusercontent.com/82134417/133857070-e7b8fe77-75b3-45c7-a4d1-40159a0bc0e5.png)

**And the winner was Gradient Boosting as it scored the lowest MAE**

In order to check how well this model predicts prices, I created a scatterplot of actual prices vs. predicted prices (see above). The closer the blue dots are to the black diagonal-dotted line, the better will be the predictions. This diagram shows that for prices less than $400, the predictions are somewhat better than for prices higher than $400. That is for price range greater than 400, most of the predictions are lower than the actual values. This must have been caused by the unbalanced data set where there’s 75% of prices are less than $156.

![image](https://user-images.githubusercontent.com/82134417/133857122-43f06e67-404a-4eeb-ad95-6e726b154869.png)  

One way to minimize this issue is to predict prices separately for different price ranges, for instance, different price model for less than $156 (price at 75th percentile), next one for prices in the range of $156 and $600 (price at 99th percentile) and another one for prices above $600.

## 3.1. Model for Prices less than $156
Train data MAE was 19.64 and test data MAE was 20.49. Solution was reached with estimators = 100 and max-depth = 5. Below is the scatterplot for comparing the actual vs. predicted prices.

![image](https://user-images.githubusercontent.com/82134417/133857180-14ec2c33-d4a4-469f-a284-b50032297577.png)  

An error analysis where the prediction error, that is, actual value minus the predicted value was compared against all the features of the sample instances. Out of 6,838 samples, around 58% of them had absolute error less than $20 and 95% of them had absolute error less than $50. For better comprehension about how each feature instances contribute to its prediction, Shapley Additive Explanations (SHAP) were used. Following SHAP summary plot shows feature importances in predicting the prices.

![image](https://user-images.githubusercontent.com/82134417/133857217-51a85480-54ab-4ba7-849b-2ae5743c8bfe.png)

method called summary_plot() from shap library offers details about feature importance and the impact of Shapley values on the prediction. Red means high feature values and blue means low feature values. So, for instance, if room_type_Entire home/apt has 1 instead of 0, it will increase the predicted price (because red values are on the right side), whereas for room_type_shared room if the value is 1 instead of 0, it will decrease the predicted price (because the red values are on the left side).

Force-plot from Shap additives provides a snapshot of which feature values have influenced the predicted values. For example, the force-plot below shows that the features in red color pushes the values towards higher log prices, whereas the blue feature values push for lower values and 4.09 will be the predicted log price for the sample.

![image](https://user-images.githubusercontent.com/82134417/133857250-73802026-8328-4dfb-87cf-59d408719856.png)  

## 3.2. Model for Prices between $156 and $600
MAE for train data was $54.18 and MAE for test data was $57.45. The scatterplot showing the actual values vs. predicted values depicts that for prices in the lower range like $156 - $350 the predictions are better. However, for prices above, $350 or so, the predictions are very much lower than the actual values.

![image](https://user-images.githubusercontent.com/82134417/133857299-c8d8be4a-20f3-45f2-afec-ce5c9097ace8.png)  

This could be because the samples are unbalanced, meaning proportionately there’s way too many lower prices than higher prices. The feature importance bar chart revealed number of bathrooms to be the most important feature in segmenting the samples. This was followed by number of accommodates and number of amenities.

![image](https://user-images.githubusercontent.com/82134417/133857349-b68fcfa5-8340-4a48-a776-6aa33f9478ea.png)  

Out of 2185 predictions, 11.4% of them have absolute error less than $10 and 61% of them had absolute error less than $50. However, 17 of them, most of their actual prices are close to $600, had absolute prediction error of $300. A shap summary plot offers information about the feature importances. Force plot depicts which features contributed towards arriving at the predicted value.

![image](https://user-images.githubusercontent.com/82134417/133857382-149148a0-2f25-48b7-8a37-51290b628bce.png)  

## 4. Limitations and Future Directions
Solution for the pricing model may be applicable only for listings in the New York area and for the first quarter of the year as the data covers the early period of April 2021. One important limitation of this study was the imbalance in data set, we can use algorithms like, SMOTER; which is a minority class oversampling technique modified for continuous target feature/regression to handle unbalanced data.

Another important issue with it is the lack of data on the demand side of the listings. One way to get data on this is to analyze the text-data on reviews of listings and find if the customers have mentioned anything specific about the location, for instance its proximity to subways or highways. Another method is to calculate how far the listings are from tourist attractions and historic places, as distance between landmarks and listings has been found to influence prices.

This project only counted the number of amenities that the listings offered, however it didn’t differentiate between the type of amenities it offered. For instance, ‘host greets you’ and ‘coffee maker’ cannot be equally compared with each other, as the former will make a significant difference in the experience.

## 5. Credits
Many thanks to my ever motivating mentor Giovanni Bruner for helping me throughout this project.
