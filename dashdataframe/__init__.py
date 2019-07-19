import dash
import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as dhc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from textwrap import dedent

__version__ = "0.0.1"

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


def configure_app(app, df,
                  link_name="dynamiclink",
                  link_func=None,
                  plot_columns=None):
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
    if plot_columns is None:
        plot_columns = df.select_dtypes(np.number).columns

    color_options = list(plot_columns)
    assert np.all(np.isin(np.array(plot_columns), df.columns))
    color_options.append('None')

    app.layout = html.Div([
        html.Div([

            html.Div([
                dcc.Dropdown(
                    id='xaxis-column',
                    options=[{'label': i, 'value': i}
                             for i in plot_columns],
                    value='depth'
                ),
                dcc.RadioItems(
                    id='xaxis-type',
                    options=[{'label': i, 'value': i}
                             for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'display': 'inline-block'}
                )
            ],
                style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    id='yaxis-column',
                    options=[{'label': i, 'value': i}
                             for i in plot_columns],
                    value='y'
                ),
                dcc.RadioItems(
                    id='yaxis-type',
                    options=[{'label': i, 'value': i}
                             for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'display': 'inline-block'}
                )
            ],
                style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    id='color_feature',
                    options=[{'label': i, 'value': i}
                             for i in color_options],
                    value='None'
                )
            ],
                style={'width': '30%', 'float': 'right', 'display': 'inline-block'}),


            html.Div([
                html.A(f'{link_name}', id='link', href='', target="_blank")
            ]),

            dcc.Graph(
                id='indicator-graphic', style={'width': 1000, 'overflowY': 'scroll'}
            ),

            html.Div([
                'Selected Points:  ',
                html.Span(' Click ', style={'color': 'red'}),
                html.Span(' Input ID ', style={
                    'color': 'orange'}),
                html.Span(' Both ', style={'color': 'orchid'}),
            ],
                style={'width': '60%', 'float': 'left', 'display': 'inline-block'}),

            html.Div([
                dcc.Markdown(dedent('''
                                    #### Where are these cells?
                                    ''')),

                dcc.Input(
                    id='id_list',
                    placeholder='Enter list of IDs you would like to highlight',
                    type='text',
                    size=100,
                    value=''
                )
            ],
                style={'width': '100%', 'display': 'inline-block'}),

            html.Div([
                html.Button('Digital Sort', id='sortButton'), html.Button(
                    'Reset Sort', id='resetButton')
            ]),
            html.Div([
                html.A('For questions or feedback, please email leilae@alleninstitute.org',
                       id='info',
                       target="_blank")
            ],
                style={'width': '40%', 'float': 'right'}),
        ])
    ])

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
        mesh_ids = []
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



