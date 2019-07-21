# DashDataFrame
An simplified interface for making dash apps to explore multi-dimensional dataframes with custom link integration and filtering. 

# Introduction
[Dash](https://plot.ly/dash/) is a powerful python framework for making interactive visualizations.  Pandas is an powerful and popular package for storing tabular data in python. DashDataFrame makes it easy to create a interactive 2d scatter plot with colored dots, that allows users to dynamically select columns to plot as x,y or colorize the points from.  Users can select data in one plot perspective, and then change the axis and points remain selected.  Users can also sub-select points using the 'digital sort' button to filter the rows of the dataframe to the points that are relevant.  Finally, by passing in a python function which creates an html link, users can integrate dash with other external viewers or files to create a custom link based upon the set of IDs the user has selected. 

# Installation
Clone this repository and install it with `python setup.py install` or `pip install dashdataframe`

# Getting Started
All you need is to get a dataframe and  dash app, and then use dashdataframe.configure_app to setup the dash app and then launch your dash app. A simpliest example..

```Python
    import dash
    from dashdataframe import configure_app
    # initialize a dash app
    app = dash.Dash()
    
    # load your dataframe from somewhere
    df = ... 
    
    # configure the app
    configure_app(app, df)
    
    # run the dash server
    app.run_server(port=8880)
```

Check out the example in (examples/test_dash.py).  You will need to pip install neuroglancer, but it will demonstrate how to create a visualization of a dataframe that summarizes basic statistics of fly brain regions, and allows you to select regions and create a custom link to visualize the selected regions in neuroglancer.   You supply configure_app with a function that takes a dataframe and a set of selected indices in your dataframe and returns a url. 

# Level of Support
We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
