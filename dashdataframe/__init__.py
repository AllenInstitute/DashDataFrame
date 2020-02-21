import dash
import importlib
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_html_components as dhc
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.figure_factory as ff
from textwrap import dedent
from typing import List
import json
import umap
import seaborn as sns
from scipy.stats import zscore

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

import dashdataframe.dashdf_layouts as dfl

#dcc = importlib.import_module("apps.dash-svm.utils.dash_reusable_components")

__version__ = "0.4.0"



# function to highlight a certain subset of points depending on selection
def highlight_points(dff, selectedData, ids, highlight_by = None):
    dff['outline_color']='white'
    dff['outline_width']=.5
    dff_highlight = dff.index
    if highlight_by != None:
        dff_highlight = dff[highlight_by]
    if selectedData:
        selected_mesh_ids = np.array([p['customdata'] for p in selectedData['points']],
                                    dtype=np.int64)
        dff.loc[dff.index.isin(selected_mesh_ids),'outline_color']='forestgreen'
        dff.loc[dff.index.isin(selected_mesh_ids),'outline_width']=4
    else:
        selected_mesh_ids =  np.array([], dtype=np.int64)
    if ids:
        list_ids = ids.split(',')
        list_ids = np.array(list_ids, dtype=np.int64)
        dff.loc[dff_highlight.isin(list_ids),'outline_width']=4
        dff.loc[dff_highlight.isin(list_ids),'outline_color']='purple'
        dff.loc[dff_highlight.isin(list_ids) & dff.index.isin(selected_mesh_ids),'outline_color']='orchid'
            
    return dff[dff.visible].outline_color.values, dff[dff.visible].outline_width.values

def calculate_new_umap(dff, metrics, znorm = False):
    M = dff.loc[dff.visible].dropna()[metrics]
    if zscore:
        M = M.apply(zscore)
        M = M.values
    embedding = umap.UMAP(min_dist=0).fit_transform(M)
    dff['e0'] = embedding[:,0]
    dff['e1'] = embedding[:,1]
    return dff

def calculate_new_kmeans(dff,num_clusters, metrics, znorm=False):
    M = dff.loc[dff.visible].dropna()[metrics]
    if zscore:
        M = M.apply(zscore)
        M = M.values
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(M)
    return kmeans.labels_

def calculate_new_aggclustering(dff, num_clusters, metrics, linkage_type, znorm=False):
    M = dff.loc[dff.visible].dropna()[metrics]
    if zscore:
        M = M.apply(zscore)
        M = M.values
    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(M)
    return cluster.labels_

def calculate_new_spectclustering(dff, num_clusters, metrics, znorm=False):
    M = dff.loc[dff.visible].dropna()[metrics]
    if zscore:
        M = M.apply(zscore)
        M = M.values
    cluster = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=0).fit_predict(M)
    return cluster.labels_



def configure_app(app, df,
                  link_name="dynamiclink",
                  link_func=None,
                  display_func=None,
                  plot_columns=None,
                  highlight_by = None,
                  add_umap=False,
                  add_clustering=False):
    """ 
    Parameters:
    df: pandas.DataFrame
        a pandas dataframe to plot
    link_func: func
       function that has the form f(indices, df) 
       where indices are the row indices of the df datafame that are selected.
       f should return a url that will be used to populate the dynamic link
    create_link: bool
        whether to create a link for each point (default = True)
    link_func: fun
        what function to call if you make a link

    """

    if plot_columns is None:
        
        plot_columns = df.select_dtypes(np.number).columns


    color_options = list(plot_columns)
    assert np.all(np.isin(np.array(plot_columns), df.columns))
    color_options.append('None')
 
    scatter_layout = dfl.make_scatter_layout(plot_columns, color_options, link_name)
    scatter_tab = dcc.Tab(label='Scatter Plot Explorer', children=[scatter_layout])
    tabs = [scatter_tab]

    if add_umap is True:
        print('adding umap tab')
        umap_layout = dfl.make_umap_layout(plot_columns, link_name)
        umap_tab = dcc.Tab(label='UMAP Explorer', children=[umap_layout])
        tabs.append(umap_tab)
    if add_clustering:
        print('adding clustering tab')
        cluster_layout = dfl.make_cluster_layout(plot_columns, link_name)
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
    app._highlight_by = highlight_by
    app._df = df
    app._df['visible'] = True
    app._df['e0'] = 0
    app._df['e1'] = 0

    @app.callback(dash.dependencies.Output('selected-data','children'),
    [dash.dependencies.Input('indicator-graphic', 'selectedData')])
    def update_selected(selectedData):
        if selectedData:
            if display_func is None:
                return []
            else:
                selected_mesh_ids = np.array([p['customdata'] for p in selectedData['points']],
                                        dtype=np.int64)
                return display_func(df, selected_mesh_ids)

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
            color = 'gray'
            min_max = [0, 1]
        else:
            color = app._df.loc[app._df.visible, color_feature]
            min_max = np.percentile(color, [0.1, 99.9])
        outline_color = ['white'] * np.sum(app._df.visible)
        outline_width = [0.5] * np.sum(app._df.visible)
        if selectedData or ids:
            outline_color, outline_width = highlight_points(app._df, selectedData, ids,
                                            highlight_by=app._highlight_by)
        hoverdata = app._df[app._df.visible].index.values

        return {
            'data': [go.Scattergl(
                x=app._df.loc[app._df.visible, xaxis_column_name],
                y=app._df.loc[app._df.visible, yaxis_column_name],
                customdata=app._df.loc[app._df.visible].index.values,
                text=hoverdata,
                mode='markers',
                marker={
                    'size': 6,
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
        dash.dependencies.Input('color_feature','value'),
        dash.dependencies.Input('umap-graphic', 'selectedData'),
        dash.dependencies.Input('umap_id_list', 'value')
        ])
        
        def update_umap_graph(selected_metrics, norm, umap_clicks, color_feature, selectedData, ids):
            
            Znorm = False
            if norm == 'Znorm':
                Znorm = True
            if umap_clicks is not None:
                if umap_clicks != app._prev_umap_clicks:
                    app._prev_umap_clicks += 1
                    app._df = calculate_new_umap(app._df,selected_metrics, znorm = Znorm)       
            # else:
            #     e0=[0]
            #     e1 = [0] 
            hoverdata = app._df[app._df.visible].index.values
            color = 'gray'
            outline_color = ['white'] * np.sum(app._df.visible)
            outline_width = [0.5] * np.sum(app._df.visible)
            if selectedData or ids:
                outline_color, outline_width = highlight_points(app._df, selectedData, ids,
                                                highlight_by=app._highlight_by)
            return {
                'data': [go.Scattergl(
                    x=app._df['e0'],
                    y=app._df['e1'],
                    customdata=app._df.loc[app._df.visible].index.values,
                    text=hoverdata,
                    mode='markers',
                    marker={
                        'line': {'width': outline_width, 'color': outline_color},
                        'size': 6,
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
            [dash.dependencies.Input('num_clusters', 'value'),
            dash.dependencies.Input('cluster-graphic', 'selectedData'),
            dash.dependencies.Input('cluster_metrics', 'values'),
            dash.dependencies.Input('cluster_norm', 'value'),
            ])
            
        def update_dendro_graph(num_clusters, selectedData,selected_metrics, norm):
            
            app._prev_cluster_clicks
            
            data = app._df.loc[app._df.visible].dropna()[selected_metrics]
            if norm == 'Znorm':
                data = data.apply(zscore)

            color_thresh = None
            if app._prev_cluster_clicks == 0:
                color_thresh = 0.0   
       
            dendro = ff.create_dendrogram(data, linkagefun= lambda x: shc.linkage(data, 'ward', metric='euclidean'), 
                                                                                    color_threshold = color_thresh)
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
                    if 'Agglomerative' in method:
                        linkage = 'single'
                        if 'Average' in method:
                            linkage = 'average'
                        if 'Ward' in method:
                            linkage = 'ward'
                        clusters = calculate_new_aggclustering(app._df, int(num_clusters), selected_metrics,linkage, znorm = znorm )
                    if method == 'KMeans':
                        clusters = calculate_new_kmeans(app._df, int(num_clusters), selected_metrics, znorm = znorm )
                    if method == 'Spectral Clustering':
                        clusters = calculate_new_spectclustering(app._df, int(num_clusters), selected_metrics, znorm = znorm  )
                    app._prev_cluster_clicks += 1
                    app._cluster_color = [palette[c] for c in clusters]

            marker_dict = {
                    'size': 7,
                    'opacity': 0.5,
                    'line': {'width': outline_width, 'color': outline_color},
                    'color': app._cluster_color
                }

            return {
                'data': [go.Scattergl(
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
            [dash.dependencies.Input('cluster-graphic', 'selectedData'),
            dash.dependencies.Input('id_list', 'value')])

        def update_cluster_link(selectedData, ids):

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


            
                        

            



