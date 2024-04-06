# Machine Learning Analysis of Hotel Reviews: Insights and Predictions
This analysis was done with Weka

## Overview of the Project

- Customer feedback can significantly influence business outcomes
- Analyzing reviews is a crucial part of understanding customer satisfaction and improving services
- This analysis is based on the dataset of TripAdvisor hotel reviews
- Aim of this analysis is to deliver actionable insights through machine learning
- Understanding these aspects can help hotels tailor their strategies to enhance customer experience and manage their online reputation

## Objective

- Develop a model that will quantify the effects of reviewers with one function presenting the effect as a function of the number of cities he visited and another from the helpfulness of his reviews. The scope of this work is to identify those reviewers who had visited more than 15 cities and had their reviews voted as helpful more than 100 times by users. From this analysis, the company can single out the experienced reviewer who significantly influences the market, thus allowing targeting marketing strategies and personalized engagements with such customers




![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/b435c261-2cbd-4a52-b181-fc14942392af)
![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/317acefa-89c5-4331-be75-3cdbf7873992)

## Overview of the Key Attributes

### Author_num_helpful_votes
![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/1d294651-0fee-42a6-a8e0-f250de8a0afc)
### Overall_rating
![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/89c17819-5787-4d11-9c89-b8b6bb3a1043)
### Author_num_cities
![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/dbd1b87b-2093-4813-9610-d8d42b8d06f3)
### Author_num_reviews
![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/1aa056be-2f96-4dd7-a8c3-a12ccc959681)

## Data Preprocessing: Dealing with missing values![image]

- There was a large amount of missing variables:
  - 9% from rating_service
  - 8% from rating_cleanliness
  - 9% from rating_value
  - 17% from rating_location
  - 39% from rating_sleep_quality
  - 14% from rating_rooms
  - 87% from rating_check_in_front_desk
  - 93% from rating_business_service_(e_g_internet_access)
  - 21% from author_num_cities
  - 15% from author_num_helpful_votes
  - 6% from author_location
- Because vast majority of variables were missing from attributes ”rating_business_service” and ”rating_check_in_front_desk” they were completely deleted
- Numerical missing variables were replaced with median variable. Filter ”ReplaceMissingValues” was used
- All the string variables were removed.

  ## Feature Engineering and Class Attribute Selection

- New attributes were created to help with the analysis
  - Nominal attributes were created to reflect whether an author has visited more than 15 cities or received more than 100 helpful votes. ”AddExpression” filter was used to create two new attributes: 
    - ”Visited_More_Than_15_cities” to indicate whether the author had visited over 15 cities or not
    - ”Over_100_Helpful_Votes” to indicate whether the author had received over 100 helpful votes or not
    - These steps were done separataly to the same dataset

## Atribute Evaluation

- Attributes ”Visited_More_Than_15_cities” and ”Over_100_Helpful_Votes” were made to be nominal variables to identify influencial factors through ”InfoGainAttributeEval” attribute evaluator where search method was ”Ranked”

- Attributes were nominally categorized as ”1” and ”0”
  - For ”Visited_More_Than_15_cities” instances with over 15 as a value were consider ”1”
  - For ”Over_100_Helpful_Votes” instances with over 100 as a value were considered ”1”

- For ”Visited_More_Than_15_cities” the most important predictors were: ”author_num_cities”, ”author_num_reviews”, ”author_num_helpful_votes”

  - ”author_num_reviews” suggests that authors who have written more reviews are more likely to have visited more cities
  - ”author_num_helpful_votes” indicates that authors whose reviews are deemed helpful also tend to travel more

- For ”Over_100_helpful_votes” the most important predictors were also: ”author_num_helpful_votes” ”author_num_cities”, ”author_num_reviews”, and ” Visited_More_Than_15_cities”

  - ”author_num_cities” suggests that the authors who travel more are more likely to provide reviews that are considered helpful
  - ”author_num_reviews” indicates that the number of reviews author has written is related to the amount they are found helpful

## Model Selection Criteria

- Ultimately, the dataset was unbalanced so SMOTE was used to balance the dataset
- Decision tree and Logistic Regression was used
- Logistic Regression is espesially well suited for nominal attributes
- Ensemble method like Random Forest could provide better performance compared to single decision tree by reducing overfitting
- Cost Matrix  was utilized to gain more information about relative importance or business impact of each type of classification error
- 10-fold cross-validation was used for all of the models

## Logistic Regression

- Logistic regression was used for the nominal variable ”over_100_helpful_votes” to predict whether a review author has received over 100 helpful votes 
- Model had overall accuracy of 87.0846%
- Classes are performing pretty similarly to eachother
- Precision and Recall are the most significantly differing
- This variable was notably more unbalanced, which leads to lower accuracy

![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/1bafd7a6-eedf-4377-96ee-e2069b3213da)

- Logistic regression was used for the nominal variable ”Visited_More_Than_15_cities” to predict whether a review author has visited over 15 cities
- Model had overall accuracy of 92.5994%
- Classes are pretty well performing compared to eachother
- Compared to logistic precision of previous variable, the accuracy is notably better

![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/c3f0fdc8-84cf-404e-ac03-9c8bc1a70e95)

## Key Findings from the Logistic Regression Models

- ”Visited_More_Than_15_cities” :The logistic regression model shows a decent capability to classify instances based on whether an author has visited more than 15 cities, but it is more reliable in predicting those who have not visited more than 15 cities than those who have. The relatively high false positive rate for predicting visits to more than 15 cities indicates a need for model refinement or consideration of additional features that could improve the prediction accuracy for this class.

- ”Over_100_Helpful_Votes” : Although logistic regression in general is highly accurate, it can be problematic in capturing the positives for the less frequent class (authors with over 100 helpful votes). This would mean that the model is generally robust but can further improve with efforts of feature engineering, rebalance techniques, or even alternative model exploration to raise its predictive performance towards identification of the influential review author by the number of his helpful votes.

## Random Forest

- Random Forest model was worse for ”over_100_helpful_votes” since this variable was more unbalanced
- This model was performing relatively well for ”visited_more_than_15_cities”

### over_100_helpful_votes
![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/3e565b18-7ef1-424f-86aa-8c658d26862a)
### visited_more_than_15_cities
![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/0161a5ce-1b25-43d4-9d8e-ba0df62407cc)

## Cost Matrix

- Cost Matrix was implemented to achieve more information about business impact of misclassification
-This was the example scenario for Cost Matrix : 
  - Sending marketing materials to a non-influential reviewer costs the company €10 (FP) but missing an influential reviewer might mean a lost opportunity cost of €100 (FN) because they could have influenced more bookings or improved brand visibility.
  - Cost matrix was structured as follows:
    - Cost of false positive: 1 (because it costs €10, this is the baseline)
    - Cost of false negative: 10 (reflecting that it's ten times worse to miss an influential reviewer)

- Decision Tree was used as the Model
  - The Model was used for both new binary variables ”over_100_helpful_votes” and ”Visited_More_Than_15_cities” to gain maximum amount of information

- The first model (visited_more_than_15_cities) has overall the worst accuracy but balanced both sensitivities in classifying both classes. The statistics for Kappa for this model show moderate agreement of predicted and actual classes, showing steady power of prediction. In addition, the moderate balance of F-Measure details how the model effectively juggles precision and recall, which is critical from the viewpoint that in such a real-life setting, misclassification might be costly. 

![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/95cfe0ac-535b-4ad3-ab04-4e7de99b13d9)

- The second model (over_100_helpful_votes) shows a worse accuracy. The lower F-Measure for this class, with the other average model performances, indicates that the model is generally reliable but may be improved for better recall, probably by employing some advanced technique like Cost Sensitive Learning or Class-Weight Adjustment. Such improvements would make the model even more sensitive to the minority class and thereby bring down the risk even further of missing influential reviewers.

![image](https://github.com/niilaikio/ML_Hotel_Reviews/assets/163767043/aed6aeda-c73c-4afd-92e9-e5e9363dc222)

## Key Insights

- The key feature predictors for influential reviews and customer satisfaction were identified to be "Author_num_reviews," "Author_num_helpful_votes," and "Author_num_cities.
- Best model for both new variables were Logistic Regression
- Worst model for both new variables were Cost Matrix with Decision Tree
- The authors with the highest number of reviews and helpful votes influence the potential customer a lot, and their satisfaction is reflected greatly in business results.
- One of the aspects on which this reviewer influence sign was noted was exhibited by the number of cities visited and its strong statistical relationship, indicative of the expectation of separate expectation of experienced travelers

## Business Recommendations

- Enhanced Reviewer Engagement: Focus on engaging with reviewers who have visited more than 15 cities and those whose reviews receive significant helpful votes. These reviewers are likely to be key influencers within the community.
- Targeted Response Strategies: Prepare focused response strategies to the feedback of the most influential reviewers. Addressing their concerns and feedback on time would, to some extent, open an opportunity that may help in increasing customer satisfaction and influencing prospective customers.
- Monitoring and Analysis: Continue to monitor and analyze the reviews with an eye on key themes related to good scores and intentions of revisits. Use this feedback for operational excellence, e.g., develop service quality and room cleanliness.
- Continuous Improvement: Leverage insights from experienced travelers to refine service offerings. 



















