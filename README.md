# TabularDataCleaner
Automating data preprocessing (cleaning) and feature selction tasks

**A. Data preprocessing (cleaning):**
  1. Drop columns with high null values (>80%)
  2. Drop columns with low information (low variance)
  3. Drop duplicate columns 
  4. Drop low cardinality columns
  5. Drop columns with high correlation (Credit to Brian Pietracatella) Original creator of this function
     
**B. Feature selection tasks**
  1. The FeatureSelector class implements feature selection using various methods like Fisher Score (ANOVA), Mutual Information, Chi-Square, and Information Value. 
  2. It ranks features based on these methods and aggregates the ranks to provide an overall ranking of the features.
