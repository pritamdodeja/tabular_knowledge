# {{{ Imports
from tabular_knowledge.tabular_knowledge import *
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
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import plotly.express as px
import scipy
# }}}
# {{{ debugging if needed
from ipdb import set_trace as ipdb
from IPython import embed as ipython
    # ipdb()
# }}}
# {{{ Data reading and variable initialization
data_file_name = '../../data/train_mod.csv'
manf_data = pd.read_csv(data_file_name)
target_label = 'f'
random_state=0
index_name = 'id'
manf_data.set_index(keys=index_name, inplace=True)
# }}}
# {{{ Data analysis
# {{{ Analyze metadata
manf_metadata = df_metadata(manf_data, numerical_threshold=40)
visualize_df_metadata(df=manf_metadata)

# The above plot is a good way to figure out the threshold for continuous vs.
# categorical variables.  It's on a log scale so it makes it much easier to
# visualize. With this, it looks like 40 might be a good threshold, below which
# we would treat that variable as categorical.  We re-run df_metadata based on
# this.  We are going to revist the variables set to non numeric later, so
# it is prudent to set the threshold kind of high

# We want to see the histograms for the data that we're thinking is not numerical
# but it has a numerical datatype
 # }}}
 # {{{ Fine tune variable types
suspected_variables = manf_metadata.query('(dtype != "object") & (is_numerical == False)').index.tolist()

# we want to use this to set some of the variables to numerical type, as they
# are incorrectly labeled as not numerical so far

visualize_histograms(df=manf_data, variables=suspected_variables, log_y=False)
# set m0, m1, m2 to numerical types

# Here we are not using numerical_threshold anymore, because we have something
# more nuanced, inspecting the distributions themselves. If there are "gaps"
# in the distributions, then I'm treating those as categorical or if they
# are categorized into a very small number of bins

change_to_numerical_list = ['m0', 'm1', 'm2']
change_variable_type_to_numerical(manf_metadata, change_to_numerical_list)

# The below should visually show m0 is now set to is_numerical
visualize_df_metadata(df=manf_metadata)           
# }}}
# {{{ Analyze mutual information
manf_mi_df, new_df = compute_mutual_information(df=manf_data, target_label=target_label, random_state=random_state, meta_df = manf_metadata, return_df=True)
# Here new_df is an encoded dataframe based off of the original data


visualize_mi_individual(df=manf_mi_df, meta_df=manf_metadata, target_label=target_label)
# Computing the mutual information is based on an estimate of 20 nearest neighbors.  How
# stable is the mutual information based on random_state is a very important thing to
# give us assurance that we are dealing with good features.
# What if we ran this algorithm a bunch of times and kept track of the rank of each
# feature.  Then we could see which features are consistently shown to have high
# mutual information and which ones vary a lot.

mi_df = mi_sampling(df=manf_data, target_label=target_label, meta_df = manf_metadata, n_neighbors=20, number_of_runs = 3, transform=True)
visualize_mi_sampling(mi_df, meta_df=manf_metadata)
# }}}
# {{{ Begin feature engineering

# Now let's get a dataframe that can serve as a starting point for our feature engineering.  This has
# the most conservative encoding and imputation. The encoding of categorical variables is dependent
# on the type of model we're going to use: Ordinal for tree based, and OneHot for Linear.  An enhancement that can be made is enrich the metadata to indicate which variables should be encoded in which way.
# Right now it just does OneHotEncoding as that's the most conservative one.

_, basic_df = compute_mutual_information(df=manf_data, target_label=target_label, random_state=random_state, meta_df = manf_metadata, return_df=True, add_indicator=True)

# Our original numerical features are still there and have the same names, we need this for the pipeline
# We are not currently doing anything with the categorical features since they are already one hot encoded
numerical_features = manf_metadata.loc[manf_metadata.is_numerical, ].index.tolist()
categorical_features = manf_metadata.loc[manf_metadata.is_numerical == False, ].index.tolist()
categorical_features_regex = [feature + '_.*' for feature in categorical_features]
categorical_features_regex = "|".join(categorical_features_regex)
missing_indicator_regex = 'missingindicator_.*'
basic_df.filter(regex=categorical_features_regex)
basic_df.filter(regex=missing_indicator_regex)
basic_df[numerical_features]
basic_df[target_label]


# Before we do feature engineering on basic_df, we can analyze it using our existing tools
numerical_threshold=10
basic_df_metadata = df_metadata(basic_df, )
visualize_df_metadata(basic_df_metadata)
suspected_variables = basic_df_metadata.query('(dtype != "object") & (is_numerical == False) & (unique_value_counts > @numerical_threshold)').index.tolist()

# we want to use this to set some of the variables to numerical type, as they
# are incorrectly labeled as not numerical so far

visualize_histograms(df=basic_df, variables=suspected_variables, log_y=False)
# set m0, m1, m2 to numerical types

# Here we are not using numerical_threshold anymore, because we have something
# more nuanced, inspecting the distributions themselves. If there are "gaps"
# in the distributions, then I'm treating those as categorical or if they
# are categorized into a very small number of bins

change_to_numerical_list = ['m0', 'm1', 'm2']
change_variable_type_to_numerical(basic_df_metadata, change_to_numerical_list)
numerical_features = basic_df_metadata.loc[basic_df_metadata.is_numerical, ].index.tolist()
categorical_features = basic_df_metadata.loc[basic_df_metadata.is_numerical == False, ].index.tolist()
categorical_features_regex = [feature + '.*' for feature in categorical_features]
categorical_features_regex = "|".join(categorical_features_regex)
missing_indicator_regex = 'missingindicator_.*'
basic_df.filter(regex=categorical_features_regex)
basic_df.filter(regex=missing_indicator_regex)
basic_df[numerical_features]
basic_df[target_label]

# Let's take basic df and scale the numerical features and pass them through a 
# Polynomial Transformer, and re-do the mutual information run.

standard_scaler = StandardScaler()
polynomial_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
numerical_preprocessor = Pipeline([
    ('standard_scaler', standard_scaler),
    ('polynomial_features', polynomial_features),
])

preprocessor = ColumnTransformer([
    ('numerical_preprocessor', numerical_preprocessor, numerical_features),
], remainder='passthrough')

basic_data_transformed = get_df(preprocessor, basic_df)
basic_data_transformed_metadata = df_metadata(basic_data_transformed)
visualize_df_metadata(basic_data_transformed_metadata)
suspected_variables = basic_data_transformed_metadata.query('(dtype != "object") & (is_numerical == False) & (unique_value_counts > @numerical_threshold)').index.tolist()

# we want to use this to set some of the variables to numerical type, as they
# are incorrectly labeled as not numerical so far

visualize_histograms(df=basic_data_transformed, variables=suspected_variables, log_y=False)
# set m0, m1, m2 to numerical types

# Here we are not using numerical_threshold anymore, because we have something
# more nuanced, inspecting the distributions themselves. If there are "gaps"
# in the distributions, then I'm treating those as categorical or if they
# are categorized into a very small number of bins

change_to_numerical_list = ['m0', 'm1', 'm2']
change_variable_type_to_numerical(basic_data_transformed_metadata, change_to_numerical_list)

basic_mi_df, basic_mi_details_df = mi_sampling(df=basic_data_transformed, target_label=target_label, meta_df = basic_data_transformed_metadata, n_neighbors=20, number_of_runs = 3, return_detailed_df=True)
visualize_mi_sampling(df=basic_mi_df, meta_df = basic_data_transformed_metadata)

visualize_mi_dispersion(basic_mi_details_df, meta_df=basic_data_transformed_metadata)
visualize_3d_sampling(df=basic_mi_df, meta_df=basic_data_transformed_metadata, n_clusters=20)
# }}}
# {{{ Identify clusters of features with good mutual information
# non api stuff below

from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
mykmeans = KMeans(n_clusters=5)
mykmeans.fit(basic_mi_df[['mean', 'std', 'median']])
basic_mi_df['cluster'] = mykmeans.predict(basic_mi_df[['mean', 'std', 'median']])
fig = px.scatter_3d(data_frame=basic_mi_df.reset_index(), x='mean', y='std', z='median'
           , log_x=True, log_y=True, log_z=True, hover_data=['index'], color='cluster')

basic_mi_df['cluster'] = mykmeans.predict(basic_mi_df)

good_features = basic_mi_df.loc[basic_mi_df['cluster'].isin([2, 3])].index.tolist()
basic_data_transformed[good_features]
# }}}
# {{{ Build a model
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(X=basic_data_transformed[good_features], y=basic_data_transformed[target_label])
# Now that we know which features are important, we need a way to construct
# a Pipeline that drops all the trash. We could just use the metadata to get
# the good feature names and drop the rest.
# }}}
# }}}
# {{{ wild west below ### #
numerical_feature_selector = feature_type_selector(dtype_include='numerical')
categorical_feature_selector = feature_type_selector(dtype_include='categorical')
numerical_feature_names = numerical_feature_selector(manf_data)
categorical_feature_names = categorical_feature_selector(manf_data)
categorical_feature_names.remove(target_label)

fig = px.bar(data_frame=manf_mi_df.drop(labels=[target_label]).reset_index(),
             x='index', y='mutual_information',)
fig.show()

fig = px.bar(data_frame=estimated_mutual_information_df.drop(labels=[target_label]).reset_index(),
             x='index', y='mutual_information',)
fig.show()

numerical_imputer = SimpleImputer(strategy='median', add_indicator=True)
categorical_imputer = SimpleImputer(strategy='most_frequent', add_indicator=True)
imputation_preprocessor = ColumnTransformer([
    ('numerical_imputer', numerical_imputer, numerical_feature_names ),
    ('categorical_imputer', categorical_imputer, categorical_feature_names ),
], remainder='passthrough')

df = manf_data.copy()
post_imputation_np = imputation_preprocessor.fit_transform(df)
feature_name_np_array = imputation_preprocessor.get_feature_names_out()
post_imputation_df = pd.DataFrame(post_imputation_np, columns=feature_name_np_array)
categorical_feature_indices = np.zeros(len(categorical_feature_names))
numerical_feature_indices = np.zeros(len(numerical_feature_names))

for position, feature in enumerate(categorical_feature_names):
    categorical_feature_indices[position] = np.where(
        feature_name_np_array == 'categorical_imputer__' + feature)[0]

for position, feature in enumerate(numerical_feature_names):
    numerical_feature_indices[position] = np.where(
        feature_name_np_array == 'numerical_imputer__' + feature)[0]

categorical_feature_indices = categorical_feature_indices.astype(
    int).tolist()
numerical_feature_indices = numerical_feature_indices.astype(int).tolist()

from sklearn.preprocessing import PolynomialFeatures
scale_and_polynomial = Pipeline([
    ('standard_scaler', StandardScaler() ),
    ('polynomial', PolynomialFeatures(degree=2, interaction_only=True)),
])
numeric_and_categorical_transformer = ColumnTransformer(
    [('OneHotEncoder', OneHotEncoder(),
      categorical_feature_indices),
     ('scale_and_polynomial', scale_and_polynomial,
      numerical_feature_indices)],
    remainder='passthrough')

post_encoding_np = numeric_and_categorical_transformer.fit_transform(post_imputation_df)
post_encoding_np_feature_names = numeric_and_categorical_transformer.get_feature_names_out()
post_encoding_df = pd.DataFrame(post_encoding_np, columns=post_encoding_np_feature_names)
post_encoding_df = post_encoding_df.rename(shorten_param, axis=1)
# }}}
