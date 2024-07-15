import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.preprocessing import LabelEncoder
from feature_engine.selection import SelectByInformationValue
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
# import eli5
import shap

class FeatureSelector:
    # For Classification problem
    def __init__(self):
        self.mi = None
        self.fisher = None
        self.chi = None
        self.iv = None
        self.FI = None
        self.shap = None
        self.PI = None
        # self.eli5 = None

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

    def FI_with_catboost(self, X, y):
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        train_pool = Pool(X, y, cat_features = categorical_features)
        if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.number):
            estimator = CatBoostRegressor(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
        else:
            estimator = CatBoostClassifier(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
        model = estimator.fit(train_pool)
        if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.number):
            estimator = CatBoostRegressor(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
        else:
            estimator = CatBoostClassifier(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
        model = estimator.fit(train_pool)   
        FI_series = pd.Series(model.get_feature_importance(train_pool,), X.columns)
        return FI_series
    
    # def FI_with_eli5(self, X, y):
    #     categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    #     train_pool = Pool(X, y, cat_features = categorical_features)
    #     if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.number):
    #         estimator = CatBoostRegressor(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
    #     else:
    #         estimator = CatBoostClassifier(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
    #     model = estimator.fit(train_pool)
    #     Series = eli5.explain_weights_catboost(catb = model,
    #                  pool = train_pool,
    #                  )
    #     Series_ = eli5.formatters.as_dataframe.format_as_dataframe(Series)
    #     eli5_series = pd.Series(data=Series_['weight'].values, index=Series_['feature'])
    #     eli5_series.name = None
    #     eli5_series.index.name = None
    #     return eli5_series

    def FI_with_shap(self, X, y):
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        train_pool = Pool(X, y, cat_features = categorical_features)
        if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.number):
            estimator = CatBoostRegressor(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
        else:
            estimator = CatBoostClassifier(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
        model = estimator.fit(train_pool)
        # Accepts only tree based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        vals = np.abs(shap_values).mean(0)
        shap_series = pd.Series(vals, X.columns)
        # shap.summary_plot(shap_values, X, plot_type='bar')
        return shap_series
    


    def P_imp(self, X, y):
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        train_pool = Pool(X, y, cat_features = categorical_features)
        if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.number):
            estimator = CatBoostRegressor(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
        else:
            estimator = CatBoostClassifier(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
        model = estimator.fit(train_pool)
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=1066)
        PI_series = pd.Series(perm_importance.importances_mean, X.columns)
        return PI_series


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

        self.iv = self.information_value(X, y)
        self.FI = self.FI_with_catboost(X, y)
        self.PI = self.P_imp(X, y)
        self.shap = self.FI_with_shap(X, y) 

        # Creating a DataFrame to compile all results
        results = pd.DataFrame(index=X.columns)
        results['Fisher Score'] = self.fisher
        results['Mutual Information'] = self.mi
        results['Chi-Square'] = self.chi
        results['Information Value'] = self.iv
        results['FI Value'] = self.FI
        results['Pimp Value'] = self.PI
        results['Shap Value'] = self.shap

        # Ranking features based on each method
        results['Fisher Rank'] = results['Fisher Score'].rank(ascending=False, method='min')
        results['MI Rank'] = results['Mutual Information'].rank(ascending=False, method='min')
        results['Chi Rank'] = results['Chi-Square'].rank(ascending=False, method='min')
        results['IV Rank'] = results['Information Value'].rank(ascending=False, method='min')
        results['FI Rank'] = results['FI Value'].rank(ascending=False, method='min')
        results['Pimp Rank'] = results['Pimp Value'].rank(ascending=False, method='min')
        results['Shap Rank'] = results['Shap Value'].rank(ascending=False, method='min')

        # Aggregating the ranks to get a combined rank
        results['Average Rank'] = results[['Fisher Rank', 'MI Rank', 'Chi Rank', 'IV Rank', "FI Rank", "Pimp Rank",
                                           'Shap Rank', 
                                           # 'Eli5 Rank'
                                          ]].mean(axis=1)
        results = results.sort_values('Average Rank')
        return results
