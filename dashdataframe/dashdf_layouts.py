from dash import dcc
from dash import html
from textwrap import dedent


def make_scatter_layout(plot_columns, color_options, link_name):
    scatter_layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(
                                "Scatter Plot Explorer", style={"color": "steelblue"}
                            ),
                            dcc.Markdown(
                                dedent(
                                    """
                ### Explore your dataset by plotting various features 
                Note: The points visible here can be clustered using the UMAP app feature. 
                If you have selected a subset of points using Filter Sort, the UMAP analysis will only include those points.
                
                """
                                )
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "inline-block",
                            "fontFamily": "Arial",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Markdown(
                                dedent(
                                    """
                Choose your axis features
                """
                                )
                            ),
                            dcc.Markdown(
                                dedent(
                                    """
                **X axis:**
                """
                                )
                            ),
                            dcc.Dropdown(
                                id="xaxis-column",
                                options=[
                                    {"label": i, "value": i} for i in plot_columns
                                ],
                                value=plot_columns[0],
                                style={"fontSize": "14"},
                            ),
                            dcc.RadioItems(
                                id="xaxis-type",
                                options=[
                                    {"label": i, "value": i} for i in ["Linear", "Log"]
                                ],
                                value="Linear",
                                labelStyle={
                                    "fontSize": "13",
                                    "display": "inline-block",
                                },
                            ),
                            dcc.Markdown(
                                dedent(
                                    """
                **Y axis:**
                """
                                )
                            ),
                            dcc.Dropdown(
                                id="yaxis-column",
                                options=[
                                    {"label": i, "value": i} for i in plot_columns
                                ],
                                value=plot_columns[1],
                                style={"fontSize": "14"},
                            ),
                            dcc.RadioItems(
                                id="yaxis-type",
                                options=[
                                    {"label": i, "value": i} for i in ["Linear", "Log"]
                                ],
                                value="Linear",
                                labelStyle={
                                    "fontSize": "13",
                                    "display": "inline-block",
                                },
                            ),
                            dcc.Markdown(
                                dedent(
                                    """
                Choose your color feature
                """
                                )
                            ),
                            dcc.Dropdown(
                                id="color_feature",
                                options=[
                                    {"label": i, "value": i} for i in color_options
                                ],
                                value="None",
                                style={"fontSize": "14"},
                            ),
                            dcc.Markdown(
                                dedent(
                                    """
                Looking for specific data points? Enter IDs below.
                """
                                )
                            ),
                            dcc.Input(
                                id="id_list",
                                placeholder="Enter list of IDs you would like to highlight",
                                type="text",
                                size="80",
                                value="",
                                style={
                                    "fontSize": "14",
                                    "height": "50px",
                                    "width": "95%",
                                },
                            ),
                            dcc.Markdown(
                                dedent(
                                    """
                To continue exploring only selected points, use Digital Sort below.
                """
                                )
                            ),
                            html.Button(
                                "Filter Sort",
                                id="sortButton",
                                style={"fontSize": "12px"},
                            ),
                            html.Button(
                                "Reset Sort",
                                id="resetButton",
                                style={"fontSize": "12px"},
                            ),
                        ],
                        style={
                            "width": "18%",
                            "display": "inline-block",
                            "height": 800,
                            "overflowY": "scroll",
                            "backgroundColor": "aliceblue",
                            "fontFamily": "Arial",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="indicator-graphic"),
                            html.A(f"{link_name}", id="link", href="", target="_blank"),
                        ],
                        style={
                            "width": "80%",
                            "float": "right",
                            "display": "inline-block",
                            "fontFamily": "Arial",
                        },
                    ),
                ]
            ),
            html.Div([html.Pre(id="selected-data")]),
        ]
    )
    return scatter_layout


def make_umap_layout(plot_columns, link_name):
    umap_layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Plot with UMAP", style={"color": "purple"}),
                            dcc.Markdown(
                                dedent(
                                    """
            ### Explore your dataset using UMAP 
            Note: The points represented are those visible in the scatter plot explorer 
            For example, if you have selected a subset using Filter Sort, this representation will only show those points.
            
            """
                                )
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "inline-block",
                            "fontFamily": "Arial",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Markdown(
                                dedent(
                                    """
            Select the features you would like to include in your UMAP plot
            """
                                )
                            ),
                            dcc.Markdown(
                                dedent(
                                    """
            Note: If certain points are missing features you have selected, they will not be included
            """
                                )
                            ),
                            dcc.Checklist(
                                id="umap_metrics",
                                options=[
                                    {"label": m, "value": m} for m in plot_columns
                                ],
                                values=[],
                                labelStyle={
                                    "fontSize": "14",
                                    "display": "inline-block",
                                },
                            ),
                            html.Button(
                                "Select All",
                                id="umap_features_button",
                                style={"fontSize": "12px"},
                            ),
                            html.Button(
                                "Reset All",
                                id="unselect_button",
                                style={"fontSize": "12px"},
                            ),
                            dcc.RadioItems(
                                id="umap_norm",
                                options=[
                                    {"label": i, "value": i}
                                    for i in ["Raw_Values", "Znorm"]
                                ],
                                value="Raw_Values",
                                labelStyle={
                                    "fontSize": "13",
                                    "display": "inline-block",
                                },
                            ),
                            html.Button(
                                "Calculate UMAP",
                                id="umap_button",
                                style={"fontSize": "12px"},
                            ),
                        ],
                        style={
                            "width": "18%",
                            "display": "inline-block",
                            "height": 800,
                            "overflowY": "scroll",
                            "backgroundColor": "ghostwhite",
                            "fontFamily": "Arial",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="umap-graphic"),
                            html.A(
                                f"{link_name}", id="umap_link", href="", target="_blank"
                            ),
                        ],
                        style={
                            "width": "80%",
                            "float": "right",
                            "display": "inline-block",
                            "fontFamily": "Arial",
                        },
                    ),
                ]
            ),
        ]
    )
    return umap_layout


def make_cluster_layout(plot_columns, link_name):

    methods = [
        "KMeans",
        "Spectral Clustering",
        "Agglomerative Ward",
        "Agglomerative Single Linkage",
        "Agglomerative Average Linkage",
    ]
    cluster_layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        className="banner",
                        children=[
                            html.H1(
                                "Hierarchical Clustering", style={"color": "green"}
                            ),
                            dcc.Markdown(
                                dedent(
                                    """
                    ### Explore your dataset using various clustering methods 
                    Note: The points clustered are those visible in scatter plot explorer 
                    For example, if you have selected a subset using the  Filter Sort button, this clustering analysis will
                    only be done on those points.
                    
                    """
                                )
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "inline-block",
                            "fontFamily": "Arial",
                        },
                    ),
                    html.Div(
                        id="body",
                        className="container scalable",
                        children=[
                            html.Div(
                                id="left_column",
                                children=[
                                    dcc.Markdown(
                                        dedent(
                                            """
                        Choose your clustering method
                        """
                                        )
                                    ),
                                    dcc.Dropdown(
                                        id="method",
                                        options=[
                                            {"label": i, "value": i} for i in methods
                                        ],
                                        value="Sklearn Agglomerative",
                                        style={"fontSize": "14"},
                                    ),
                                    dcc.Input(
                                        id="num_clusters",
                                        placeholder="Enter the number of clusters yo",
                                        type="text",
                                        size="80",
                                        value=None,
                                        style={
                                            "fontSize": "14",
                                            "height": "50px",
                                            "width": "95%",
                                        },
                                    ),
                                    dcc.Markdown(
                                        dedent(
                                            """
                        Select the features you would like to include in your analysis
                        """
                                        )
                                    ),
                                    dcc.Markdown(
                                        dedent(
                                            """
                        Note: If certain rows are missing features you have selected, the rows will not be included
                        """
                                        )
                                    ),
                                    dcc.Checklist(
                                        id="cluster_metrics",
                                        options=[
                                            {"label": m, "value": m}
                                            for m in plot_columns
                                        ],
                                        values=[],
                                        labelStyle={
                                            "fontSize": "14",
                                            "display": "inline-block",
                                        },
                                    ),
                                    html.Button(
                                        "Select All",
                                        id="cluster_features_button",
                                        # style={'fontSize':'12px'}
                                    ),
                                    html.Button(
                                        "Reset All",
                                        id="reset_features_button",
                                        # style={'fontSize':'12px'}
                                    ),
                                    dcc.RadioItems(
                                        id="cluster_norm",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in ["Raw_Values", "Znorm"]
                                        ],
                                        value="Raw_Values",
                                        # labelStyle={'fontSize':'13','display': 'inline-block'}
                                    ),
                                    html.Button(
                                        "Cluster!",
                                        id="clusterButton",
                                        # style={'fontSize':'12px'}
                                    ),
                                    dcc.Markdown(
                                        dedent(
                                            """
                        Choose X and Y axis features for the scatter plot
                        """
                                        )
                                    ),
                                    dcc.Markdown(
                                        dedent(
                                            """
                        **X axis:**
                        """
                                        )
                                    ),
                                    dcc.Dropdown(
                                        id="cluster-xaxis-column",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in plot_columns
                                        ],
                                        value=plot_columns[0],
                                        style={"fontSize": "14"},
                                    ),
                                    dcc.RadioItems(
                                        id="cluster-xaxis-type",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in ["Linear", "Log"]
                                        ],
                                        value="Linear",
                                        labelStyle={
                                            "fontSize": "13",
                                            "display": "inline-block",
                                        },
                                    ),
                                    dcc.Markdown(
                                        dedent(
                                            """
                        **Y axis:**
                        """
                                        )
                                    ),
                                    dcc.Dropdown(
                                        id="cluster-yaxis-column",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in plot_columns
                                        ],
                                        value=plot_columns[1],
                                        style={"fontSize": "14"},
                                    ),
                                    dcc.RadioItems(
                                        id="cluster-yaxis-type",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in ["Linear", "Log"]
                                        ],
                                        value="Linear",
                                        labelStyle={
                                            "fontSize": "13",
                                            "display": "inline-block",
                                        },
                                    ),
                                ],
                                style={
                                    "width": "18%",
                                    "display": "inline-block",
                                    "overflowY": "scroll",
                                    "height": 800,
                                    "backgroundColor": "honeydew",
                                    "fontFamily": "Arial",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="dendro-graphic"),
                                    dcc.Graph(id="cluster-graphic"),
                                    html.A(
                                        f"{link_name}",
                                        id="cluster_link",
                                        href="",
                                        target="_blank",
                                    ),
                                ],
                                style={
                                    "width": "80%",
                                    "float": "right",
                                    "display": "inline-block",
                                    "fontFamily": "Arial",
                                },
                            ),
                        ],
                    ),
                ]
            )
        ]
    )
    return cluster_layout
