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
    metadata_frame['dtype'] = metadata_frame['dtype'].astype('string')
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

def compute_mutual_information(df, target_label, meta_df, random_state, return_df=False, add_indicator=False, transform=True, n_neighbors=10):
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
        categorical_imputer = SimpleImputer(strategy='most_frequent', add_indicator=add_indicator)
        categorical_pipeline = Pipeline([
            ('categorical_imputer', categorical_imputer),
            ('OneHotEncoder', OneHotEncoder()),

    ])
        imputation_preprocessor = ColumnTransformer(
            [('numerical_imputer',
              SimpleImputer(strategy='median', add_indicator=add_indicator),
              numerical_features),
         ('categorical_pipeline',
          categorical_pipeline,
          categorical_features)
         ],
            remainder='passthrough')

        # We need to figure out the indices to the features that are supposed to be scaled and encoded by the next
        # step

        post_imputation_np = imputation_preprocessor.fit_transform(df)
        feature_name_np_array = imputation_preprocessor.get_feature_names_out()
        categorical_feature_indices = np.zeros(len(categorical_features))
        numerical_feature_indices = np.zeros(len(numerical_features))

        # for position, feature in enumerate(categorical_features):
        #     categorical_feature_indices[position] = np.where(
        #         feature_name_np_array == 'categorical_pipeline__' + feature)[0]

        for position, feature in enumerate(numerical_features):
            numerical_feature_indices[position] = np.where(
                feature_name_np_array == 'numerical_imputer__' + feature)[0]

        categorical_feature_indices = categorical_feature_indices.astype(
            int).tolist()
        numerical_feature_indices = numerical_feature_indices.astype(int).tolist()

        numeric_and_categorical_transformer = ColumnTransformer(
            [('OneHotEncoder', OneHotEncoder(),
              []),
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
        X=df_transformed, y=df[target_label].astype(int), random_state=random_state, n_neighbors=n_neighbors)
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
# {{{ mi_sampling
def mi_sampling(df, target_label, meta_df, n_neighbors, number_of_runs=2, transform=False):
    transformed_mi_list = []
    for seed in range(number_of_runs):
        summarized_mi_df = compute_mutual_information(df=df, target_label=target_label, random_state=seed, meta_df = meta_df, return_df=False, add_indicator=False, transform=transform)
        summarized_mi_df.columns = [f'mutual_information_run_{seed}']
        transformed_mi_list.append(summarized_mi_df)
    merged_mi_df = transformed_mi_list[0].copy()
    for run in range(1, number_of_runs):
        merged_mi_df = merged_mi_df.merge(transformed_mi_list[run], left_index=True, right_index=True)
    summarized_mi_df = pd.DataFrame()
    summarized_mi_df['mean'] = merged_mi_df.T.mean()
    summarized_mi_df['std'] = merged_mi_df.T.std()
    return summarized_mi_df
# }}}
# {{{ Visualize mi sampling
def visualize_mi_sampling(df):
    fig = px.scatter(data_frame=df.reset_index(), x='mean', y='std', log_x=True, log_y=True, color='index')
    fig.show()
    # }}}
# {{{ visualize_df_metadata
def visualize_df_metadata(df, x='variable', y='unique_value_counts', color='is_numerical'):
    fig = px.bar(data_frame=df,
                 x=x, y=y, log_y=True, color=color)
    fig.show()
    # }}}
# {{{ Visualize histograms
def visualize_histograms(df, variables, log_y=True):
    if len(variables) == 1:
        fig = px.histogram(data_frame=df[[variables]], facet_col='variable', log_y=log_y, )
    else:
        fig = px.histogram(data_frame=df[variables], facet_col='variable', log_y=log_y, )
    fig.update_xaxes(matches=None)
    # fig.update_yaxes(matches=None)
    fig.show()
# }}}
# {{{ change variables to numerical
def change_variable_type_to_numerical(meta_df, change_to_numerical_list):
    meta_df.loc[meta_df['variable'].isin(change_to_numerical_list), 'is_numerical'] = True
    # }}}
# {{{ Visualize individual mi
def visualize_mi_individual(df, target_label, x='index'):
    fig = px.bar(data_frame=df.drop(labels=[target_label]).reset_index(),
                 x=x, y='mutual_information',)
    fig.show()
# }}}
if __name__ == '__main__':
    pass
