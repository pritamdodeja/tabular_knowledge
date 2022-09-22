# {{{ Imports
import seaborn.objects as so
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import plotly.express as px
import scipy
# }}}
# # {{{ Names and file reading
# AMES_DATA_PATH = "/home/pdodeja/programming/learning/data/datasets/ames/train.csv"
# PIMA_DATA_PATH = "/home/pdodeja/programming/learning/data/datasets/pima-indians/diabetes.csv"
# random_state = 0

# ames_data = pd.read_csv(filepath_or_buffer=AMES_DATA_PATH)
# pima_data = pd.read_csv(filepath_or_buffer=PIMA_DATA_PATH)

# ames_data.head()
# pima_data.head()
# # }}}
# {{{ df_metadata
def df_metadata(df, numerical_threshold=50):
    list_of_variables = df.columns.tolist()
    list_of_dtypes = [df.dtypes[variable] for variable in list_of_variables]
    categorical_selector = selector(dtype_include=object)
    numerical_selector = selector(dtype_exclude=object)
    unique_value_counts = [df[variable].nunique()
                           for variable in list_of_variables]
    categorical_features = categorical_selector(df)
    numerical_features = numerical_selector(df)
    is_numerical_init = [True] * len(list_of_variables)
    metadata_frame = pd.DataFrame(
        {'variable': list_of_variables, 'dtype': list_of_dtypes,
         'is_numerical': is_numerical_init,
         'unique_value_counts': unique_value_counts})
    null_sum = df.isnull().sum()
    null_sum.name = 'null_sum'
    metadata_frame = pd.merge(
        metadata_frame,
        null_sum,
        left_on='variable',
        right_index=True)
    metadata_frame['samples_missing'] = metadata_frame['null_sum'] > 0
    total_samples = len(df)
    metadata_frame['percent_missing'] = metadata_frame['null_sum'] / total_samples
    for feature in categorical_features:
        metadata_frame.loc[metadata_frame.variable ==
                           feature, ['is_numerical']] = False
    for feature in numerical_features:
        if df[feature].nunique() < numerical_threshold:
            # print(f"Updating feature {feature}")
            metadata_frame.loc[metadata_frame.variable ==
                               feature, ['is_numerical']] = False
    return metadata_frame
# }}}
# {{{ feature_type_selector
def feature_type_selector(dtype_include=None):
    def nested_function(df,):
        meta_df = df_metadata(df)
        if dtype_include == 'numerical':
            return meta_df.loc[meta_df.is_numerical, 'variable'].tolist()
        else:
            return meta_df.loc[meta_df.is_numerical ==
                               False, 'variable'].tolist()
    return nested_function
# }}}
# {{{ shorten_param function
def shorten_param(param_name):
    if "__" in param_name:
        if len(param_name.rsplit(" ", 1)) < 2:
            return param_name.rsplit("__", 1)[1]
        else:
            return str(shorten_param(param_name.rsplit(" ", 1)[
                       0])) + " " + shorten_param(' '.join(param_name.rsplit(" ", 1)[1:]))
    return param_name
# }}}
# {{{ Compute mutual information - takes as input target label and original df
# returns dataframe with mutual information

# Algorithm

# Start out with original data
# Analyze it to determine data types
# Encode it per determined types
# Compute mutual information
# Return mutual information

def compute_mutual_information(df, target_label, meta_df, random_state, return_df=False, add_indicator=False, transform=True):
    # Analyze data frame
    # meta_df = df_metadata(df, numerical_threshold=numerical_threshold)
    target_is_numerical = meta_df.loc[meta_df.variable == target_label][
        'is_numerical'].iloc[0]

    # Determine problem type
    if target_is_numerical:
        problem_type = 'regression'
        mutual_information_function = mutual_info_regression
    else:
        problem_type = 'classification'
        mutual_information_function = mutual_info_classif

    # Select feature types
    numerical_features = meta_df.loc[meta_df.is_numerical, 'variable'].tolist()
    categorical_features = meta_df.loc[meta_df.is_numerical == False, 'variable'].tolist()

    # Remove target label from features
    for feature_list in [numerical_features, categorical_features]:
        if target_label in feature_list:
            feature_list.remove(target_label)

    # Transform df
    if transform:
        imputation_preprocessor = ColumnTransformer(
            [('numerical_imputer',
              SimpleImputer(strategy='median', add_indicator=add_indicator),
              numerical_features),
             ('categorical_imputer',
              SimpleImputer(strategy='most_frequent', add_indicator=add_indicator),
              categorical_features)],
            remainder='passthrough')

        # We need to figure out the indices to the features that are supposed to be scaled and encoded by the next
        # step

        post_imputation_np = imputation_preprocessor.fit_transform(df)
        feature_name_np_array = imputation_preprocessor.get_feature_names_out()
        categorical_feature_indices = np.zeros(len(categorical_features))
        numerical_feature_indices = np.zeros(len(numerical_features))

        for position, feature in enumerate(categorical_features):
            categorical_feature_indices[position] = np.where(
                feature_name_np_array == 'categorical_imputer__' + feature)[0]

        for position, feature in enumerate(numerical_features):
            numerical_feature_indices[position] = np.where(
                feature_name_np_array == 'numerical_imputer__' + feature)[0]

        categorical_feature_indices = categorical_feature_indices.astype(
            int).tolist()
        numerical_feature_indices = numerical_feature_indices.astype(int).tolist()

        numeric_and_categorical_transformer = ColumnTransformer(
            [('OneHotEncoder', OneHotEncoder(),
              categorical_feature_indices),
             # ('StandardScaler', StandardScaler(),
             #  numerical_feature_indices)
             ],
            remainder='passthrough')
        preprocessor = Pipeline(
            [('imputation_preprocessor', imputation_preprocessor),
             ('numeric_and_categorical_transformer',
              numeric_and_categorical_transformer)])
        df_transformed_np = preprocessor.fit_transform(df)
        preprocessed_feature_names = list(preprocessor.get_feature_names_out())
        if isinstance(df_transformed_np, scipy.sparse._csr.csr_matrix):
            df_transformed = pd.DataFrame(
                df_transformed_np.todense(),
                columns=preprocessed_feature_names)
        else:
            df_transformed = pd.DataFrame(
                df_transformed_np,
                columns=preprocessed_feature_names)
        df_transformed = df_transformed.rename(shorten_param, axis=1)
    else:
        df_transformed = df.copy()
        preprocessed_feature_names = df.columns.tolist()
    target_series = df[target_label]
    estimated_mutual_information = mutual_information_function(
        X=df_transformed, y=df[target_label].astype(int), random_state=random_state, n_neighbors=10)
    estimated_mutual_information_df = pd.DataFrame(
        estimated_mutual_information.T.reshape(
            1, -1), columns=preprocessed_feature_names)
    estimated_mutual_information_df = estimated_mutual_information_df.rename(
        shorten_param,
        axis=1)
    estimated_mutual_information_df = estimated_mutual_information_df.T
    estimated_mutual_information_df.columns = ['mutual_information']
    # estimated_mutual_information_df = estimated_mutual_information_df.sort_values(
    #     by=['mutual_information'])
    if return_df:
        return estimated_mutual_information_df, df_transformed
    else:
        return estimated_mutual_information_df
# }}}
# {{{ Selector based on mutual information


def mi_selector(mi_threshold=0.05, target_label=None, random_state=0):
    def selector_to_return(df,):
        mi_df = compute_mutual_information(
            df=df,
            target_label=target_label,
            random_state=random_state)
        matching_variables = mi_df[mi_df.loc[:,'mutual_information'] > mi_threshold].index.tolist()
        matching_features = []
        # Remove target
        if target_label in matching_variables:
            matching_variables.remove(target_label)
        # Only return from features that were in original df
        # since we compute more than that as we impute, encode etc.
        for feature_name in df.columns.tolist():
            if feature_name in matching_variables:
                matching_features.append(feature_name)
        return matching_features
    return selector_to_return
# }}}
# {{{ get_df
def get_df(pipe, data):
    new_data = pipe.fit_transform(data)
    if isinstance(new_data, scipy.sparse._csr.csr_matrix):
        print(type(new_data))
        return pd.DataFrame(
            new_data.todense(),
            columns=pipe.get_feature_names_out()).rename(
            shorten_param, axis=1)
    else:
        return pd.DataFrame(
            new_data,
            columns=pipe.get_feature_names_out()).rename(
            shorten_param,
            axis=1)
    # }}}
if __name__ == '__main__':
    pass
