# Tabular Knowledge

## Goals of package

Make it easy to analyze tabular datasets with lots of features in scikit-learn.

### Motivation

Although scikit-learn has selectors that let you select features based on their
data types, sometimes you want to select features based on mutual information.
In order to do this, metadata needs to be computed, similar to how tensorflow
extended, and specifically, tensorflow data validation does it.  On top of that
mutual information needs be computed for each feature to determine which 
features have predictive power for the problem at hand.  

Tabular knowledge makes it easier to understand data in a scikit-learn/pandas
environment and provides functions to make it easy to filter out features
that do not have any predictive power.


A working example of its use is available in src/client/client.py which 
implements a typical workflow that can lead to a baseline model.  Specifically,
computing metadata, fine tuning semantics of features, analyzing mutual 
information, basic encoding with pipelines, followed by a model. 

### Future directions

It may be possible to build policy driven pipelines in the future.  For example
include features with mutual information above a certain threshold, where
the selector behaves like make_column_selector in scikit-learn.  Secondly,
the encoding could also be declarative based on model type.

