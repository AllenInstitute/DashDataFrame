# DashDataFrame
An simplified interface for making dash apps to explore multi-dimensional dataframes with custom link integration and filtering. 

# Introduction
[Dash](https://plot.ly/dash/) is a powerful python framework for making interactive visualizations.  Pandas is an powerful and popular package for storing tabular data in python. DashDataFrame makes it easy to create a interactive 2d scatter plot with colored dots, that allows users to dynamically select columns to plot as x,y or colorize the points from.  Users can select data in one plot perspective, and then change the axis and points remain selected.  Users can also sub-select points using the 'digital sort' button to filter the rows of the dataframe to the points that are relevant.  Finally, by passing in a python function which creates an html link, users can integrate dash with other external viewers or files to create a custom link based upon the set of IDs the user has selected. 

# Getting Started

