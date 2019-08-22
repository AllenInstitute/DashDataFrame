import dash
import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as dhc
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.figure_factory as ff
from textwrap import dedent
from typing import List

import umap
import seaborn as sns
from scipy.stats import zscore

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering, KMeans


__version__ = "0.2.0"

# function to highlight a certain subset of points depending on selection
def highlight_points(dff, selectedData, ids):
        
    dff['outline_color']='white'
    dff['outline_width']=.5
    if selectedData:
        selected_mesh_ids = np.array([p['customdata'] for p in selectedData['points']],
                                    dtype=np.int64)
        dff.loc[dff.index.isin(selected_mesh_ids),'outline_color']='firebrick'
        dff.loc[dff.index.isin(selected_mesh_ids),'outline_width']=4
    else:
        selected_mesh_ids =  np.array([], dtype=np.int64)
    if ids:
        list_ids = ids.split(',')
        list_ids = np.array(list_ids, dtype=np.int64)
        dff.loc[dff.index.isin(list_ids),'outline_color']='orange'
        dff.loc[dff.index.isin(list_ids) & dff.index.isin(selected_mesh_ids),'outline_color']='orchid'
            
    return dff[dff.visible].outline_color.values, dff[dff.visible].outline_width.values

def calculate_new_umap(dff, metrics, znorm = False):
    M = dff.loc[dff.visible].dropna()[metrics]
    if zscore:
        M = M.apply(zscore)
        M = M.values
    embedding = umap.UMAP(min_dist=0).fit_transform(M)
    
    return embedding[:,0],embedding[:,1]

def calculate_new_kmeans(dff,num_clusters, metrics, znorm=False):
    M = dff.loc[dff.visible].dropna()[metrics]
    if zscore:
        M = M.apply(zscore)
        M = M.values
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(M)
    return kmeans.labels_

def calculate_new_aggclustering(dff, num_clusters, metrics, znorm=False):
    M = dff.loc[dff.visible].dropna()[metrics]
    if zscore:
        M = M.apply(zscore)
        M = M.values
    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(M)
    return cluster.labels_


def configure_app(app, df,
                  link_name="dynamiclink",
                  link_func=None,
                  plot_columns=None,
                  add_umap=False,
                  add_clustering=False):
    """ 
    Parameters:
    df: pandas.DataFrame
        a pandas dataframe to plot
    link_func: func
       what text to display on the link generated for  call the link
       function should take a list of row indices in 
    create_link: bool
        whether to create a link for each point (default = True)
    link_func: fun
        what function to call if you make a link

    """
    print(link_name,link_func,plot_columns,add_umap,add_clustering)
    if plot_columns is None:
        
        plot_columns = df.select_dtypes(np.number).columns


    color_options = list(plot_columns)
    assert np.all(np.isin(np.array(plot_columns), df.columns))
    color_options.append('None')
    

    scatter_layout = html.Div([
        html.Div([

            html.Div([
                html.H1('Scatter Plot Explorer',
                       style={'color':'steelblue'}),
                dcc.Markdown(dedent('''
                    ### Explore your dataset by plotting various features 
                    Note: The points visible here can be clustered using the UMAP app feature. 
                    If you have selected a subset of points using Filter Sort, the UMAP analysis will only include those points.
                    
                    ''')),
    
            ],
            style={'width': '100%', 'display': 'inline-block', 'fontFamily':'Arial'}),
            
            html.Div([
                dcc.Markdown(dedent('''
                    Choose your axis features
                    ''')),
                dcc.Markdown(dedent('''
                    **X axis:**
                    ''')),
                dcc.Dropdown(
                    id='xaxis-column',
                    options=[{'label': i, 'value': i} for i in plot_columns],
                    value=plot_columns[0],
                    style={'fontSize':'14'}
                ),
                dcc.RadioItems(
                    id='xaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'fontSize':'13','display': 'inline-block'}
                ),
                dcc.Markdown(dedent('''
                    **Y axis:**
                    ''')),
                dcc.Dropdown(
                    id='yaxis-column',
                    options=[{'label': i, 'value': i} for i in plot_columns],
                    value=plot_columns[1],
                    style={'fontSize':'14'}
                ),
                dcc.RadioItems(
                    id='yaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'fontSize':'13', 'display': 'inline-block'}
                ),
                dcc.Markdown(dedent('''
                    Choose your color feature
                    ''')),
                dcc.Dropdown(
                    id='color_feature',
                    options=[{'label': i, 'value': i} for i in color_options],
                    value='None',
                    style={'fontSize':'14'}
                ),
                
                dcc.Markdown(dedent('''
                    Looking for specific data points? Enter IDs below.
                    ''')),
    
                dcc.Input(
                    id = 'id_list',
                    placeholder='Enter list of IDs you would like to highlight',
                    type='text',
                    size = 80,
                    value='',
                    style= {'fontSize': '14', 'height': '50px', 'width':'95%'}
                ),
                
                dcc.Markdown(dedent('''
                    To continue exploring only selected points, use Digital Sort below.
                    ''')),
                html.Button('Filter Sort', id='sortButton', style={'fontSize':'12px'}), 
                
                html.Button('Reset Sort', id='resetButton', style={'fontSize':'12px'})
            ],
            style={'width': '18%', 'display': 'inline-block', 'height':'600px', 
                           'backgroundColor':'aliceblue', 'fontFamily':'Arial'}),
            
            
            html.Div([  
                dcc.Graph(id='indicator-graphic'),
                
                html.A(f'{link_name}', id='link', href='', target="_blank")
                ],
           
                style={'width': '80%','float':'right', 'display':'inline-block', 'fontFamily':'Arial'}),
    
        ]),
    ])

    umap_layout = html.Div([
    html.Div([

        html.Div([
            html.H1('Plot with UMAP',
                style={'color':'purple'}),
            dcc.Markdown(dedent('''
            ### Explore your dataset using UMAP 
            Note: The points represented are those visible in the scatter plot explorer 
            For example, if you have selected a subset using Filter Sort, this representation will only show those points.
            
            ''')),

            ],
            style={'width': '100%', 'display': 'inline-block', 'fontFamily':'Arial'}),

        html.Div([
            dcc.Markdown(dedent('''
            Select the features you would like to include in your UMAP plot
            ''')),
            dcc.Markdown(dedent('''
            Note: If certain points are missing features you have selected, they will not be included
            ''')),

            dcc.Checklist(
            id = 'umap_metrics',
            options = [{'label': m, 'value':m} for m in plot_columns],

            values=[],
            labelStyle={'fontSize': '14', 'display': 'inline-block'}
            ),

            html.Button('Select All', 
                    id='umap_features_button',
                    style={'fontSize':'12px'}),

            html.Button('Reset All', 
                    id='unselect_button',
                    style={'fontSize':'12px'}),

            dcc.RadioItems(
                id='umap_norm',
                options=[{'label': i, 'value': i} for i in ['Raw_Values','Znorm']],
                value='Raw_Values',
                labelStyle={'fontSize':'13','display': 'inline-block'}
                ),

            html.Button('Calculate UMAP', 
                    id='umap_button',
                    style={'fontSize':'12px'})

        ],
        style={'width': '18%', 'display': 'inline-block', 'height':'600px', 
                'backgroundColor':'ghostwhite', 'fontFamily':'Arial'}),

        html.Div([

            dcc.Graph(id='umap-graphic'),

            html.A(f'{link_name}', id='umap_link', href='', target="_blank")
        ],
        style={'width': '80%','float':'right', 'display':'inline-block', 'fontFamily':'Arial'}),

        ]),
    ])
    

    methods = ["KMeans", "Sklearn Agglomerative"]
 
    cluster_layout = html.Div([
        html.Div([

            html.Div([
                html.H1('Hierarchical Clustering',
                    style={'color':'green'}),
                dcc.Markdown(dedent('''
                ### Explore your dataset using various clustering methods 
                Note: The points clustered are those visible in scatter plot explorer 
                For example, if you have selected a subset using the  Filter Sort button, this clustering analysis will
                only be done on those points.
                
                ''')),

                ],
            style={'width': '100%', 'display': 'inline-block', 'fontFamily':'Arial'}),

            html.Div([
                dcc.Markdown(dedent('''
                Choose your clustering method
                ''')),
                dcc.Dropdown(
                    id='method',
                    options=[{'label': i, 'value': i} for i in methods],
                    value='Sklearn Agglomerative',
                    style={'fontSize':'14'}
                    ),

                dcc.Input(
                    id = 'num_clusters',
                    placeholder='Enter the number of clusters yo',
                    type='text',
                    size = 80,
                    value=None,
                    style= {'fontSize': '14', 'height': '50px', 'width':'95%'}
                    ),
                dcc.Markdown(dedent('''
                Select the features you would like to include in your analysis
                ''')),
                dcc.Markdown(dedent('''
                Note: If certain rows are missing features you have selected, the rows will not be included
                ''')),
            
                dcc.Checklist(
                    id = 'cluster_metrics',
                    options = [{'label': m, 'value':m} for m in plot_columns],

                    values=[],
                    labelStyle={'fontSize': '14', 'display': 'inline-block'}
                    ),

                html.Button('Select All', 
                        id='cluster_features_button',
                        style={'fontSize':'12px'}),

                html.Button('Reset All', 
                    id='reset_features_button',
                    style={'fontSize':'12px'}),

                dcc.RadioItems(
                    id='cluster_norm',
                    options=[{'label': i, 'value': i} for i in ['Raw_Values','Znorm']],
                    value='Raw_Values',
                    labelStyle={'fontSize':'13','display': 'inline-block'}
                    ),

                html.Button('Cluster!', 
                        id='clusterButton',
                        style={'fontSize':'12px'}),

                dcc.Markdown(dedent('''
                Choose X and Y axis features for the scatter plot
                ''')),
                dcc.Markdown(dedent('''
                **X axis:**
                ''')),
                dcc.Dropdown(
                    id='cluster-xaxis-column',
                    options=[{'label': i, 'value': i} for i in plot_columns],
                    value=plot_columns[0],
                    style={'fontSize':'14'}
                    ),
                dcc.RadioItems(
                    id='cluster-xaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'fontSize':'13','display': 'inline-block'}
                    ),
                dcc.Markdown(dedent('''
                **Y axis:**
                ''')),
                dcc.Dropdown(
                    id='cluster-yaxis-column',
                    options=[{'label': i, 'value': i} for i in plot_columns],
                    value=plot_columns[1],
                    style={'fontSize':'14'}
                    ),
                dcc.RadioItems(
                    id='cluster-yaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'fontSize':'13','display': 'inline-block'}
                    )],
            style={'width': '18%', 'display': 'inline-block', 'height':'1000px', 
                    'backgroundColor':'honeydew', 'fontFamily':'Arial'}),

            html.Div([

                dcc.Graph(id='dendro-graphic'),

                dcc.Graph(id='cluster-graphic'),

                html.A('Neuroglancer',id= 'cluster_link', 
                        href='', 
                        target="_blank"),
                ],
            style={'width': '80%','float':'right', 'display':'inline-block', 'fontFamily':'Arial'}),

        ])
    ])
    scatter_tab = dcc.Tab(label='Scatter Plot Explorer', children=[scatter_layout])
    tabs = [scatter_tab]

    if add_umap is True:
        print('true umap')
        umap_tab = dcc.Tab(label='UMAP Explorer', children=[umap_layout])
        tabs.append(umap_tab)
    if add_clustering:
        print('true clustering')
        cluster_tab = dcc.Tab(label='Clustering Explorer', children=[cluster_layout])
        tabs.append(cluster_tab)

    app.layout = html.Div([
    dcc.Tabs(id="tabs", children=tabs,
        style={'fontSize':'14','fontFamily':'Arial'})
])

    app._prev_umap_feat_clicks = 0
    app._prev_umap_clicks = 0
    app._prev_umap_none_clicks = 0
    app._cluster_color = 'steelblue'
    app._prev_cluster_clicks = 0
    app._prev_cluster_feat_clicks = 0
    app._prev_cluster_none_clicks = 0
    app._prev_sort_clicks = 0
    app._prev_reset_clicks = 0
    app._df = df
    app._df['visible'] = True

    @app.callback(
        dash.dependencies.Output('indicator-graphic', 'figure'),
        [dash.dependencies.Input('xaxis-column', 'value'),
        dash.dependencies.Input('yaxis-column', 'value'),
        dash.dependencies.Input('xaxis-type', 'value'),
        dash.dependencies.Input('yaxis-type', 'value'),
        dash.dependencies.Input('color_feature', 'value'),
        dash.dependencies.Input('sortButton', 'n_clicks'),
        dash.dependencies.Input('resetButton', 'n_clicks'),
        dash.dependencies.Input('indicator-graphic', 'selectedData'),
        dash.dependencies.Input('id_list', 'value')
    ])

    def update_graph(xaxis_column_name, yaxis_column_name,
                    xaxis_type, yaxis_type, color_feature, sort_clicks, reset_clicks,
                    selectedData, ids):

        if sort_clicks is not None:
            if sort_clicks != app._prev_sort_clicks:
                app._prev_sort_clicks += 1
                selected_ids = np.array(
                    [p['customdata'] for p in selectedData['points']], dtype=np.int64)
                app._df['visible'] = app._df.index.isin(selected_ids)
                selectedData = None
        if reset_clicks is not None:
            if reset_clicks != app._prev_reset_clicks:
                app._prev_reset_clicks += 1
                app._df.visible = True
                selectedData = None

        if color_feature == 'None':
            color = 'mediumpurple'
            min_max = [0, 1]
        else:
            color = app._df.loc[app._df.visible, color_feature]
            min_max = np.percentile(color, [0.1, 99.9])
        outline_color = ['white'] * np.sum(app._df.visible)
        outline_width = [0.5] * np.sum(app._df.visible)
        if selectedData or ids:
            outline_color, outline_width = highlight_points(app._df, selectedData, ids)
        hoverdata = app._df[app._df.visible].index.values

        return {
            'data': [go.Scatter(
                x=app._df.loc[app._df.visible, xaxis_column_name],
                y=app._df.loc[app._df.visible, yaxis_column_name],
                customdata=app._df.loc[app._df.visible].index.values,
                text=hoverdata,
                mode='markers',
                marker={
                    'size': 5,
                    'opacity': 0.5,
                    'line': {'width': outline_width, 'color': outline_color},
                    'color': color,
                    'cmax': min_max[1],
                    'cmin': min_max[0],
                    'colorscale': 'Viridis',
                    'colorbar': {'title': color_feature}
                }
            )],
            'layout': go.Layout(
                xaxis={
                    'title': xaxis_column_name,
                    'type': 'linear' if xaxis_type == 'Linear' else 'log'
                },
                yaxis={
                    'title': yaxis_column_name,
                    'type': 'linear' if yaxis_type == 'Linear' else 'log'
                },
                height = 600,
                margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                hovermode='closest',
                clickmode='event+select',
                legend={'x': 0, 'y': 1}
            )
        }


    @app.callback(dash.dependencies.Output('link', 'href'),
                [dash.dependencies.Input('indicator-graphic', 'selectedData'),
                dash.dependencies.Input('id_list', 'value')])

    def update_link(selectedData, ids):
        
        if selectedData:
            selected_mesh_ids = np.array([p['customdata'] for p in selectedData['points']],
                                        dtype=np.int64)
            selected_df = app._df[app._df.index.isin(selected_mesh_ids)]
            selected_mesh_ids = selected_df.index.values
        else:
            selected_mesh_ids = np.array([], dtype=np.int64)
        if ids:
            list_ids = np.array(ids.split(','), dtype=np.int64)
            selected_mesh_ids = np.concatenate([selected_mesh_ids, list_ids])
        if link_func is not None:
            link = link_func(selected_mesh_ids, app._df)
        else:
            link = ""
        return link
    

    if add_umap:
        @app.callback(
        dash.dependencies.Output('umap-graphic', 'figure'),
        [dash.dependencies.Input('umap_metrics', 'values'),
        dash.dependencies.Input('umap_norm', 'value'),
        dash.dependencies.Input('umap_button', 'n_clicks'),
        dash.dependencies.Input('color_feature','value')
        ])
        
        def update_umap_graph(selected_metrics, norm, umap_clicks, color_feature):
            
            Znorm = False
            if norm == 'Znorm':
                Znorm = True
            if umap_clicks is not None:
                if umap_clicks != app._prev_umap_clicks:
                    app._prev_umap_clicks += 1
                    e0, e1 = calculate_new_umap(app._df,selected_metrics, znorm = Znorm)       
            else:
                e0=[0]
                e1 = [0] 
            hoverdata = app._df[app._df.visible].index.values
            color = 'purple'
            return {
                'data': [go.Scatter(
                    x=e0,
                    y=e1,
                    customdata=app._df.loc[app._df.visible].index.values,
                    text=hoverdata,
                    mode='markers',
                    marker={
                        'size': 7,
                        'opacity': 0.5,
                        'color': color
                    }
                )],
                'layout': go.Layout(
                    xaxis={
                        'title': 'Umap axis 0',
                    },
                    yaxis={
                        'title': 'Umap axis 1',
                    },
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                    hovermode='closest',
                    height=600,
                    clickmode = 'event+select',
                    legend = {'x': 0, 'y': 1}
                )
            }

        @app.callback(dash.dependencies.Output('umap_metrics', 'values'),
            [dash.dependencies.Input('umap_metrics','options'),
             dash.dependencies.Input('umap_features_button','n_clicks'),
             dash.dependencies.Input('unselect_button', 'n_clicks')
            ])
        def select_umap_features(options, all_clicks, none_clicks):
      
            if all_clicks is not None:
                if all_clicks != app._prev_umap_feat_clicks:
                    app._prev_umap_feat_clicks +=1
                    return [i['value'] for i in options]
            
            if none_clicks is not None:
                if none_clicks != app._prev_umap_none_clicks:
                    app._prev_umap_none_clicks +=1
                    return []
                
            raise PreventUpdate()

        @app.callback(dash.dependencies.Output('umap_link', 'href'),
            [dash.dependencies.Input('umap-graphic', 'selectedData'),
            dash.dependencies.Input('id_list', 'value')])
        def update_umap_link(selectedData, ids):
            
            if selectedData:
                selected_mesh_ids = np.array([p['customdata'] for p in selectedData['points']],
                                            dtype=np.int64)
                selected_df = app._df[app._df.index.isin(selected_mesh_ids)]
                selected_mesh_ids = selected_df.index.values
            else:
                selected_mesh_ids = np.array([], dtype=np.int64)
            if ids:
                list_ids = np.array(ids.split(','), dtype=np.int64)
                selected_mesh_ids = np.concatenate([selected_mesh_ids, list_ids])
            if link_func is not None:
                link = link_func(selected_mesh_ids, app._df)
            else:
                link = ""
            return link


    if add_clustering ==True:
        @app.callback(dash.dependencies.Output('dendro-graphic', 'figure'),
            [dash.dependencies.Input('clusterButton', 'n_clicks'),
            dash.dependencies.Input('num_clusters', 'value'),
            dash.dependencies.Input('cluster-graphic', 'selectedData'),
            dash.dependencies.Input('cluster_metrics', 'values'),
            dash.dependencies.Input('cluster_norm', 'value'),
            ])
            
        def update_dendro_graph(cluster_clicks,num_clusters, selectedData,selected_metrics, norm):
            
            app._prev_cluster_clicks
            
            data = app._df.loc[app._df.visible].dropna()[selected_metrics]
            if norm == 'Znorm':
                data = data.apply(zscore)
            
                
            #palette = sns.color_palette(palette='Set2').as_hex()
            
            # if num_clusters is not None:
            #     colors = [palette[c] for c in range(int(num_clusters))]
            #shc.set_link_color_palette(list(palette.as_hex()))
            dendro = ff.create_dendrogram(data, linkagefun= lambda x: shc.linkage(data, 'ward', metric='euclidean'))
            dendro['layout'].update({'height':600, 'xaxis': {'automargin': True, 'showticklabels':False}})
            return dendro
        
        @app.callback(
            dash.dependencies.Output('cluster-graphic', 'figure'),
            [dash.dependencies.Input('method', 'value'),
            dash.dependencies.Input('num_clusters', 'value'),
            dash.dependencies.Input('cluster-xaxis-column', 'value'),
            dash.dependencies.Input('cluster-yaxis-column', 'value'),
            dash.dependencies.Input('cluster-xaxis-type', 'value'),
            dash.dependencies.Input('cluster-yaxis-type', 'value'),
            dash.dependencies.Input('cluster_metrics', 'values'),
            dash.dependencies.Input('clusterButton', 'n_clicks'),
            dash.dependencies.Input('cluster-graphic', 'selectedData'),
            dash.dependencies.Input('cluster_norm', 'value'),
            ])
        
        def update_cluster_graph(method, num_clusters, xaxis_column_name, yaxis_column_name,
                        xaxis_type, yaxis_type, selected_metrics, cluster_clicks,
                        selectedData, norm):
            
            app._cluster_color
            app._prev_cluster_clicks
        
            

            outline_color = ['white'] * np.sum(app._df.visible)
            outline_width = [0.5] * np.sum(app._df.visible)

            if selectedData:
                outline_color, outline_width = highlight_points(app._df, selectedData, id_type)

            hoverdata = app._df[app._df.visible].index.values

            znorm = False
            if norm=='Znorm':
                znorm = True

            palette = sns.color_palette(palette='Set2').as_hex()
            if cluster_clicks is not None:
                if cluster_clicks != app._prev_cluster_clicks:
                    if method == 'Sklearn Agglomerative':
                        clusters = calculate_new_aggclustering(app._df, int(num_clusters), selected_metrics, znorm = znorm )
                    if method == 'KMeans':
                        clusters = calculate_new_kmeans(app._df, int(num_clusters), selected_metrics, znorm = znorm )
                    app._prev_cluster_clicks += 1
                    app._cluster_color = [palette[c] for c in clusters]

            marker_dict = {
                    'size': 7,
                    'opacity': 0.5,
                    'line': {'width': outline_width, 'color': outline_color},
                    'color': app._cluster_color
                }

            return {
                'data': [go.Scatter(
                    x=app._df.loc[app._df.visible, xaxis_column_name],
                    y=app._df.loc[app._df.visible, yaxis_column_name],
                    customdata=app._df.loc[app._df.visible].index.values,
                    text=hoverdata,
                    mode='markers',
                    marker=marker_dict
                )],
                'layout': go.Layout(
                    xaxis={
                        'title': xaxis_column_name,
                        'type': 'linear' if xaxis_type == 'Linear' else 'log'
                    },
                    yaxis={
                        'title': yaxis_column_name,
                        'type': 'linear' if yaxis_type == 'Linear' else 'log'
                    },
                    height = 600,
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                    hovermode='closest',
                    clickmode = 'event+select',
                    legend = {'x': 0, 'y': 1}
                )
            }
        @app.callback(dash.dependencies.Output('cluster_metrics', 'values'),
             [dash.dependencies.Input('cluster_metrics','options'),
              dash.dependencies.Input('cluster_features_button','n_clicks'),
              dash.dependencies.Input('reset_features_button','n_clicks')
             ])

        def select_cluster_features(options, all_clicks, none_clicks):
      
            if all_clicks is not None:
                if all_clicks != app._prev_cluster_feat_clicks:
                    app._prev_cluster_feat_clicks +=1
                    return [i['value'] for i in options]
            
            if none_clicks is not None:
                if none_clicks != app._prev_cluster_none_clicks:
                    app._prev_cluster_none_clicks +=1
                    return []
                
            raise PreventUpdate()
        

        @app.callback(dash.dependencies.Output('cluster_link', 'href'),
            [dash.dependencies.Input('cluster-graphic', 'selectedData')
            ])

        def update_cluster_link(selectedData):

            if selectedData:
                selected_mesh_ids = np.array([p['customdata'] for p in selectedData['points']],
                                            dtype=np.int64)
                selected_df = app._df[app._df.index.isin(selected_mesh_ids)]
                selected_mesh_ids = selected_df.index.values
            else:
                selected_mesh_ids = np.array([], dtype=np.int64)
            if ids:
                list_ids = np.array(ids.split(','), dtype=np.int64)
                selected_mesh_ids = np.concatenate([selected_mesh_ids, list_ids])
            if link_func is not None:
                link = link_func(selected_mesh_ids, app._df)
            else:
                link = ""
            return link


            
                        

            



