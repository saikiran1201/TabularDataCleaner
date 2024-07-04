import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.preprocessing import LabelEncoder
from feature_engine.selection import SelectByInformationValue

class FeatureSelector:
    def __init__(self):
        self.mi = None
        self.fisher = None
        self.chi = None
        self.iv = None

    def mutual_information(self, X, y):
        mi = mutual_info_classif(X, y)
        mi_series = pd.Series(mi, index=X.columns, name="Mutual Information")
        mi_series = mi_series.sort_values(ascending=False)
        return mi_series

    def chi_square(self, X, y):
        chi, _ = chi2(X, y)
        chi_series = pd.Series(chi, index=X.columns, name="Chi-Square")
        chi_series = chi_series.sort_values(ascending=False)
        return chi_series

    def fisher_score(self, X, y):
        F, _ = f_classif(X, y)
        fisher_score_series = pd.Series(F, index=X.columns, name="Fisher Score")
        fisher_score_series = fisher_score_series.sort_values(ascending=False)
        return fisher_score_series

    def information_value(self, X, y):
        iv = SelectByInformationValue()
        iv.fit(X, y)
        iv_series = pd.Series(iv.information_values_, index=X.columns, name="Information Value")
        iv_series = iv_series.sort_values(ascending=False)
        return iv_series

    def fit(self, X, y):
        print("Calculating Fisher Score for numerical features...")
        
        # Separate numerical and categorical features
        numerical_features = X.select_dtypes(include=[np.number])
        categorical_features = X.select_dtypes(exclude=[np.number])

        if not numerical_features.empty:
            self.fisher = self.fisher_score(numerical_features, y)
        else:
            self.fisher = pd.Series([], name="Fisher Score")

        if not categorical_features.empty:
            print("Calculating Mutual Information and Chi-Square for categorical features...")
            # Encode categorical variables
            X_encoded = categorical_features.apply(LabelEncoder().fit_transform)
            self.mi = self.mutual_information(X_encoded, y)
            self.chi = self.chi_square(X_encoded, y)
        else:
            self.mi = pd.Series([], name="Mutual Information")
            self.chi = pd.Series([], name="Chi-Square")

        # Calculate Information Value
        self.iv = self.information_value(X, y)

        # Creating a DataFrame to compile all results
        results = pd.DataFrame(index=X.columns)
        results['Fisher Score'] = self.fisher
        results['Mutual Information'] = self.mi
        results['Chi-Square'] = self.chi
        results['Information Value'] = self.iv

        # Ranking features based on each method
        results['Fisher Rank'] = results['Fisher Score'].rank(ascending=False, method='min')
        results['MI Rank'] = results['Mutual Information'].rank(ascending=False, method='min')
        results['Chi Rank'] = results['Chi-Square'].rank(ascending=False, method='min')
        results['IV Rank'] = results['Information Value'].rank(ascending=False, method='min')

        # Aggregating the ranks to get a combined rank
        results['Average Rank'] = results[['Fisher Rank', 'MI Rank', 'Chi Rank', 'IV Rank']].mean(axis=1)
        results = results.sort_values('Average Rank')

        return results

