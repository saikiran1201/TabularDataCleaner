from itertools import chain
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import shap


class dataprepare():
    def __init__(self, df, nulls = 0.8, corr_limit = 0.8, var_limit = 0.01, 
                    cardinality_limit = 1, target = None):
        
        self.df = df
        self.target = target
        self.nulls = nulls  # percent of nulls in the columns > nulls, column will be droped
        self.corr_limit = corr_limit
        self.var_limit = var_limit
        self.cardinality_limit= cardinality_limit
        self.cat_columns = df.select_dtypes(include=['category', object, bool]).columns.to_list()
        #self.cat_columns = df.columns[ (df.nunique(dropna=False) < len(df) // 100) & (df.nunique(dropna=False) > 2) ]
        self.num_columns = df.select_dtypes(exclude=['category', object]).columns.to_list()
        # Dictionary to hold removal operations
        self.FI = {}
        self.ops = {}
        

    def drop_high_nulls(self, df, nulls = 0.8):
        drop_col = [i for i in df.columns if df[i].isnull().mean() >= nulls]
        print(f'High nulls columns with nulls percent greater than equals to {nulls} % in columns {drop_col} with length {len(drop_col)}')
        self.ops['drop_high_nulls'] = drop_col
        return drop_col

    # def drop_duplicate_rows(self, df):
    #     df = df.drop_duplicates()
    #     return df

    def variance_thsld(self, df, var_limit = 0.01):
        df_num = df[self.num_columns].copy()
        pipe = Pipeline([('selector', VarianceThreshold(var_limit))])
        _ = pipe.fit_transform(df_num)
        drop_col = list(set(df_num.columns) - set(pipe.get_feature_names_out()))
        print(f'Low variance columns that have less varaiance than {var_limit} with columns {drop_col} with length {len(drop_col)}')
        self.ops['variance_thsld'] = list(set(df_num.columns) - set(pipe.get_feature_names_out()))
        return list(set(df_num.columns) - set(pipe.get_feature_names_out()))


    def drop_duplicate_columns(self, df):
        list1 = df.columns
        df1 = df.T.drop_duplicates().T.copy()
        list2 = df1.columns
        drop_col = list(set(list1) - set(list2))
        print(f'Duplicate columns with same data {drop_col} with length {len(drop_col)}')
        self.ops['drop_duplicate_columns'] = list(set(list1) - set(list2))
        return list(set(list1) - set(list2))   # Doesnt return the same column names if df have two or more same col names


    # Original creater Brian Pietracatella
    # https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6
    # Thank to Brian Pietracatella

    def calcDrop(res): 
        all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))
        
        poss_drop = list(set(res['drop'].tolist()))

        keep = list(set(all_corr_vars).difference(set(poss_drop)))
        
        p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
        q = list(set(p['v1'].tolist() + p['v2'].tolist()))
        drop = (list(set(q).difference(set(keep))))

        poss_drop = list(set(poss_drop).difference(set(drop)))
        
        m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
            
        more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
        for item in more_drop:
            drop.append(item)
            
        return drop

    def drop_high_corr(self, df, corr_limit = 0.8):
        corr_mtx = df.corr().abs()
        avg_corr = corr_mtx.mean(axis = 1)
        up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(bool))
        
        dropcols = list()
        
        res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 
                                    'v2.target','corr', 'drop' ]))
        
        for row in range(len(up)-1):
            col_idx = row + 1
            for col in range (col_idx, len(up)):
                if(corr_mtx.iloc[row, col] > corr_limit):
                    if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                        dropcols.append(row)
                        drop = corr_mtx.columns[row]
                    else: 
                        dropcols.append(col)
                        drop = corr_mtx.columns[col]
                    
                    s = pd.Series([corr_mtx.index[row],
                    up.columns[col],
                    avg_corr[row],
                    avg_corr[col],
                    up.iloc[row,col],
                    drop],
                    index = res.columns)
            
                    res.loc[len(res)] = s
        
        dropcols_names = dataprepare.calcDrop(res)
        #df = df.drop(dropcols_names, axis = 1)
        print(f'Columns with high correlation {dropcols_names} with length {len(dropcols_names)}')
        self.ops['drop_high_corr'] = dropcols_names
        return dropcols_names

    def drop_low_cardinality(self, df, cardinality_limit = 1):
        drop_crdnty_col = df.columns[df.nunique() <= cardinality_limit].to_list()
        self.ops['drop_low_cardinality'] = drop_crdnty_col
        print(f'Columns with cardinality equals to {cardinality_limit} are {drop_crdnty_col} with length {len(drop_crdnty_col)}')
        return drop_crdnty_col

    def FI_with_shap(self, model, X_test):
        
        # Accepts only tree based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        vals = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame(list(zip(list(X_test.columns), vals)), columns=['features','importance'])
        importance_df.sort_values(by=['importance'], ascending=False,inplace=True)
        shap.summary_plot(shap_values, X_test, plot_type='bar')
            # Normalize the feature importances to add up to one
        importance_df['normalized_importance'] = importance_df['importance'] / importance_df['importance'].sum()
        importance_df['cumulative_importance'] = np.cumsum(importance_df['normalized_importance'])
        drop = importance_df[importance_df['cumulative_importance'] > 0.99]['features'].to_list()
        return importance_df, drop
