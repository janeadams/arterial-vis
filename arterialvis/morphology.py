"""Analytics, transformation, and rendering for visualization of morphology and network embedding

Functions:
    get_files -> array
    filter_files -> array
    
    get_edgelist -> dataframe
    generate_graph -> NetworkX graph object
    convert_to_edgelist -> dataframe
    simplifyGraph -> NetworkX graph object
    
    get_2d_positions -> tuple
    get_2d_traces -> tuple
    get_3d_traces -> tuple
    
    build_compund_graph -> Plotly figure object
    build_grouped_graph -> Plotly figure object
    build_comparison_dashboard
    
    normalize_series -> array of floats
    
    reimport_newXYZ -> dataframe
    generate_inter_edgelist -> array of dataframes
    
    extract_real_abstract -> array of dataframes
    build_animation -> Plotly figure object
"""

import os
from os import walk
from pathlib import Path
from dotenv import dotenv_values
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import networkx as nx
import traceback
from sklearn.preprocessing import normalize
import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_files(path=None):
    """Get a list of filenames from the location where .swc files are stored
    
    Keyword arguments:
        path -- (str) (optional) where to search for files; if blank, use location specified in .env
    
    Returns: an array of filenames
    """
    if (path==None):
        config = dotenv_values(".env")
        path = Path(config['SWC_SAVE'])
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files

def filter_files(files, colored=True):
    """Filter out only .swc files that are/are not colored from the sample data
    
    Keyword arguments:
        files -- (array) a list of filenames
        colored -- (bool) whether to include or exclude colored files
    
    Returns: An array of file names
    """
    filtered_files = []
    for filename in files:
        if colored and ("ColorCoded" in filename):
                filtered_files.append(filename)
        if not colored and ("ColorCoded" not in filename):
                filtered_files.append(filename)
    return filtered_files

def get_edgelist(file, path=None, output=False):
    """Extract an edgelist from the .swc morphology data
    
    Keyword arguments:
        file -- (str) the name of the file to analyze
        path -- (str or path) (optional) where to search for the file; if blank, use location specified in .env
        output -- (str or path) (optional) where to cache
    
    Returns: Dataframe edgelist
    """
    if output:
        try:
            edgelist = pd.read_csv(os.path.join(output,'edgelist.csv'))
            return edgelist
        except:
            pass
    if (path==None):
        config = dotenv_values(".env")
        path = Path(config['SWC_SAVE'])
    edgelist = pd.read_csv(os.path.join(path,file),
                     delim_whitespace=True,
                     header=None,
                     skiprows=1).rename(columns={0:'source',
                                                 1:'color',
                                                 6:'target',
                                                 2:'x',
                                                 3:'y',
                                                 4:'z',
                                                 5:'length'})
    if output:
        try: os.makedirs(output)
        except: pass
        edgelist.to_csv(os.path.join(output,'edgelist.csv'), index=False)
    return edgelist

def generate_graph(edgelist, output=False):
    """Generate a NetworkX graph from the edgelist
    
    Keyword arguments:
        edgelist -- (dataframe) the extracted edgelist containing information about edges
        output -- (str or path) (optional) where to cache
        
    Returns: a NetworkX graph object
    """
    if output:
        try: os.makedirs(output)
        except: pass
        try:
            G = pd.read_pickle(os.path.join(output,'graph.pkl'))
            print(f'Found saved file at {os.path.join(output,"graph.pkl")}')
            return G
        except:
            pass
    G=nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr='length')
    if 'color' in list(edgelist.columns):
        nx.set_node_attributes(G, pd.Series(edgelist['color'], index=edgelist['source']).to_dict(), 'group')
    if output:
        try: os.makedirs(output)
        except: pass
        with open(os.path.join(output,'graph.pkl'), 'wb') as f:
            pickle.dump(G, f)
    return G

def convert_to_edgelist(graph):
    """Convert a NetworkX graph to a dataframe edgelist
    
    Keyword arguments:
        graph -- (object) a NetworkX graph object
    
    Returns: A dataframe edgelist
    """
    return nx.to_pandas_edgelist(graph)

def simplifyGraph(G, output=False):
    """Loop over the graph until all nodes of degree 2 have been removed and their incident edges fused
    
    Source: https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges
    Authors: mjkvaak, louis_guitton, and sauce_interstellaire
    
    Keyword arguments:
        G -- (object) a NetworkX graph object
        
    Returns: a NetworkX graph object
    """
    
    if output:
        try:
            try: os.makedirs(output)
            except: pass
            g = pd.read_pickle(os.path.join(output,'simplified_graph.pkl'))
            print(f'Found saved file at {output}.pkl')
            return g
        except:
            pass
    
    g = G.copy()

    while any(degree==2 for _, degree in g.degree):

        g0 = g.copy() 
        for node, degree in g.degree():
            if degree==2:
                edges = g0.edges(node, data=True)
                edges = list(edges.__iter__())
                a0,b0,attr0 = edges[0]
                a1,b1,attr1 = edges[1]
                e0 = a0 if a0!=node else b0
                e1 = a1 if a1!=node else b1

                g0.remove_node(node)
                g0.add_edge(e0, e1, length=attr0['length']+attr1['length'])
        g = g0
    if output:
        try: os.makedirs(output)
        except: pass
        with open(os.path.join(output,'simplified_graph.pkl'), 'wb') as f:
            pickle.dump(g, f)
    return g

def get_2d_positions(G, edgelist, layout=nx.kamada_kawai_layout, output=False):
    """Obtain 2D positioning information based on a NetworkX layout algorithm
    
    Keyword arguments:
        G -- a NetworkX graph object
        edgelist -- (dataframe) the extracted edgelist containing information about edges
        layout -- (object) the NetworkX graph layout algorithm to use (default: nx.kamada_kawai_layout)
        output -- (str or path) (optional) where to cache
    
    Returns: Tuple of dictionaries (edges, nodes)
    """
    
    if output:
        try:
            edges = pd.read_pickle(os.path.join(output,'edges.pkl'))
            print(f'Found saved file at {os.path.join(output,"edges.pkl")}')
            nodes = pd.read_pickle(os.path.join(output,'nodes.pkl'))
            print(f'Found saved file at {os.path.join(output,"nodes.pkl")}')
            return edges, nodes
        except:
            pass
    
    pos=layout(G, weight='length')
    
    # Save x, y locations of each edge
    edge_x = []
    edge_y = []

    # Calculate x,y positions of an edge's 'start' (x0,y0) and 'end' (x1,y1) points
    for edge in G.edges():
        if edge[0] in list(edgelist['source']):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_y.append(y0)
            edge_y.append(y1)

    # Bundle it all up in a dict:
    edges = dict(x=edge_x,y=edge_y, z=[0]*len(edge_x))

    # Save x, y locations of each node
    node_x = []
    node_y = []

    # Save node stats for annotation
    node_name = []
    node_groups = []

    # Calculate x,y positions of nodes
    for node in G.nodes():
        if node in list(edgelist['source']):
            node_name.append(node)# Save node names
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            try:
                node_groups.append(edgelist.loc[edgelist['source']==node]['color'].item())
            except:
                pass

    # Bundle it all up in a dict:
    nodes = dict(
        x=node_x,
        y=node_y,
        z=[0]*len(node_x),
        name=node_name
    )
    if len(node_groups) == len(node_x) == len(node_y):
        nodes['groups'] = node_groups
    if output:
        try: os.makedirs(output)
        except: pass
        with open(os.path.join(output, 'edges.pkl'), 'wb') as f:
            pickle.dump(edges, f)
        with open(os.path.join(output, 'nodes.pkl'), 'wb') as f:
            pickle.dump(nodes, f)
    return edges, nodes

def get_2d_traces(G, edgelist, nodeColor=None, edgeColor=None, nodesize=1, layout=nx.kamada_kawai_layout):
    """Obtain traces for plotting 2D positions in Plotly on a NetworkX layout algorithm
    
    Keyword arguments:
        G -- a NetworkX graph object
        edgelist -- (dataframe) the extracted edgelist containing information about edges
        nodeColor -- (str) the color of the nodes, e.g. 'red'
        edgeColor -- (str) the color of the edges, e.g. 'blue'
        nodeSize -- (int) 
        layout -- (object) the NetworkX graph layout algorithm to use (default: nx.kamada_kawai_layout)
    
    Returns: Tuple of dictionaries (edge_trace, node_trace)
    """
    
    edges, nodes = get_2d_positions(G, edgelist, layout=nx.kamada_kawai_layout)

    edge_trace=go.Scatter(x=edges['x'],
                   y=edges['y'],
                   mode='lines',
                   line=dict(color='gray',
                             width=3),
                   hoverinfo='none',
                            opacity=0.3
                   )

    node_trace=go.Scatter(x=nodes['x'],
                   y=nodes['y'],
                   mode='markers',
                   name='actors',
                   marker=dict(symbol='circle',
                                 size=nodesize,
                                 color=nodes['groups'] if nodeColor==None else nodeColor,
                               colorscale='Viridis'
                                 ),
                   hoverinfo='text'
                   )

    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )
    return edge_trace, node_trace

def get_3d_traces(G, edgelist, nodeColor=None, edgeColor=None, colorScale='Viridis', nodeSize=1):
    """Generate 3D traces for Plotly Graphs
    
    Keyword arguments:
        G -- a NetworkX graph object
        edgelist -- (dataframe) the extracted edgelist containing information about edges
        nodeColor -- (str) the color of the nodes, e.g. 'red'
        edgeColor -- (str) the color of the edges, e.g. 'blue'
        colorScale -- (str) the color scale to use for groups, if present
        nodeSize -- (int) 
    
    Returns: Tuple of dictionaries (edges, nodes)
    """
    edge_x = []
    edge_y = []
    edge_z = []
    edge_group = []
    
    for edge in G.edges():

        if ((edge[0] in list(edgelist['source'])) & (edge[1] in list(edgelist['source']))):

            x0, y0, z0 = edgelist.loc[edgelist['source']==edge[0]]['x'].item(), \
            edgelist.loc[edgelist['source']==edge[0]]['y'].item(), \
            edgelist.loc[edgelist['source']==edge[0]]['z'].item()

            x1, y1, z1 = edgelist.loc[edgelist['source']==edge[1]]['x'].item(), \
            edgelist.loc[edgelist['source']==edge[1]]['y'].item(), \
            edgelist.loc[edgelist['source']==edge[1]]['z'].item()
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)

    node_x = []
    node_y = []
    node_z = []
    node_group = []
    
    for node in G.nodes():
        if node in list(edgelist['source']):
            x, y, z = edgelist.loc[edgelist['source']==node]['x'].item(),edgelist.loc[edgelist['source']==node]['y'].item(),edgelist.loc[edgelist['source']==node]['z'].item()
            group = edgelist.loc[edgelist['source']==node]['color'].item()
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_group.append(group)

    edge_trace=go.Scatter3d(x=edge_x,
                   y=edge_y, z=edge_z,
                   mode='lines',
                   line=dict(color = edge_group  if edgeColor==None else edgeColor,
                             colorscale=colorScale,
                             width=3),
                   hoverinfo='none',
                            opacity=0.3
                   )

    node_trace=go.Scatter3d(x=node_x,
                   y=node_y,
                   z=node_z,
                   mode='markers',
                   name='actors',
                   marker=dict(symbol='circle',
                                 size=nodeSize,
                                 color= node_group if nodeColor==None else nodeColor,
                                colorscale=colorScale
                                 )
                   )

    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )
    return edge_trace, node_trace

def draw_graph(traces, title='Network Graph', output=False):
    """Generate a graph from traces
    
    Keyword arguments:
        traces -- (tuple, array) a tuple or list of traces
        title -- (str) the title of the graph
        output -- (str or path) where to export graph
    
    Returns: A Plotly figure object
    """
    fig = go.Figure(data=traces,
             layout=go.Layout(
                 template='plotly_white',
                title=title,
                titlefont_size=16,
                #showlegend=False,
                hovermode='closest',
                height=800,
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    if output:
        try: os.makedirs(output)
        except: pass
        fig.write_html(os.path.join(output,'graph.html'))
    return fig

def build_compound_graph(file, path=None, output=False):
    """Generate a compound graph (showing simple vs. complete morphology) from a file
    
    Keyword arguments:
        file -- (str) the filename
        path -- (str or path) (optional) where to search for the file; if blank, use location specified in .env
        output -- (str or path) where to export graph
    
    Returns: A Plotly figure object
    """
    if (path==None):
        config = dotenv_values(".env")
        path = Path(config['SWC_SAVE'])
    edgelist = get_edgelist(file, path)
    g = generate_graph(edgelist)
    sparse = simplifyGraph(g)
    traces = []
    main_traces = get_3d_traces(g, edgelist, nodeColor='blue', edgeColor='blue')
    sparse_traces = get_3d_traces(sparse, edgelist, nodeColor='red', edgeColor='red', nodeSize=5)
    traces.extend(main_traces)
    traces.extend(sparse_traces)
    fig = draw_graph(traces, title=file)
    if output:
        try: os.makedirs(output)
        except: pass
        fig.write_html(os.path.join(output,'compound_graph.html'))
    return fig

def build_grouped_graph(file, path=None, output=False):
    """Generate a grouped graph (showing simple vs. complete morphology) from a file
    
    Keyword arguments:
        file -- (str) the filename
        path -- (str or path) (optional) where to search for the file; if blank, use location specified in .env
        output -- (str or path) where to export graph
    
    Returns: A Plotly figure object
    """
    if (path==None):
        config = dotenv_values(".env")
        path = Path(config['SWC_SAVE'])
    edgelist = get_edgelist(file, path)
    g = generate_graph(edgelist)
    sparse = simplifyGraph(g)
    traces = get_3d_traces(sparse, edgelist, edgeColor='gray', nodeSize=5)
    fig=draw_graph(traces, title=file)
    if output:
        try: os.makedirs(output)
        except: pass
        fig.write_html(os.path.join(output,'grouped_graph.html'))
    return fig

def build_comparison_dashboard(path=None):
    """Start a development server with a dashboard app showing the compound versus grouped graph for all morphology files in the specified path
    
    Keyword arguments:
        path -- (str or path) (optional) where to search for the file; if blank, use location specified in .env
    
    Returns: A Plotly figure object
    """
    if (path==None):
        config = dotenv_values(".env")
        path = Path(config['SWC_SAVE'])
    files = get_files(path)
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': f.split('_')[0], 'value': f'{f}'} for f in files],
                value=f'{files[0]}'
            ),
            html.Div(id='dd-output-container')
        ]),
        html.Div([
            dcc.Graph(id='overlay', figure=build_compound_graph(files[0], path))
        ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'middle'}),
        html.Div([
            dcc.Graph(id='grouped', figure=build_grouped_graph(files[0], path))
        ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'middle'})
    ])


    @app.callback(
        [Output('overlay', 'figure'),Output('grouped', 'figure')],
        Input('file-dropdown', 'value')
    )
    def update_output(file):
        return build_compound_graph(file, path), build_grouped_graph(file, path)
    
    app.run_server(debug=True, port=7770, threaded=True, use_reloader=False)

def normalize_series(series):
    """Apply linear algebra to normalize a series
    
    Keyword arguments:
        series -- (array) the series of numbers to normalize
    
    Returns: an array of normalized numbers
    """
    norm1 = series / np.linalg.norm(series)
    norm2 = normalize(series[:,np.newaxis], axis=0).ravel()
    return norm2

def reimport_newXYZ(edgelist, G, layout=nx.kamada_kawai_layout, z=0.5, output=False):
    """Convert to 2D, then generate an edgelist in 3D with Z-values set to a static number 
    
    Keyword arguments:
        edgelist -- (dataframe) the extracted edgelist containing information about edges
        G -- (object) a NetworkX graph object
        layout -- (object) the NetworkX graph layout algorithm to use (default: nx.kamada_kawai_layout)
        z -- (int) the static value of the 2D Z-values (default: 0.5)
        output -- (str or path) (optional) where to cache
        
    Returns: a dataframe of edges with _real and _abstract (X, Y, Z) values
    """
    new_edges, new_nodes = get_2d_positions(G, edgelist, layout=layout)
    newnodeDF = pd.DataFrame(new_nodes)
    newnodeDF['z']=z
    extended_edgelist = edgelist.merge(newnodeDF, left_on='source', right_on='name', suffixes=('_real', '_abstract'))
    
    if output:
        try: os.makedirs(output)
        except: pass
        extended_edgelist.to_csv(os.path.join(output, 'extended_edgelist.csv'), index=False)
    return extended_edgelist

def generate_inter_edgelist(sourceEdgelist, nsteps=100, output=False):
    """Generate an array of dataframes for each step of the animation
    
    Keyword arguments:
        sourceEdgelist -- (dataframe)  the extracted edgelist containing real *and* abstract X,Y,Z locations
        nsteps -- (int) the number of steps for linear interpolation
        output -- (str or path) (optional) where to cache
        
    Returns: an array of dataframes
    """
    interEdges = []
    for i in np.arange(1,nsteps):
        intermediary_edgelist = sourceEdgelist.copy()
        
        
        # Real + ( ( Real - Abstract ) / StepN ) * i
        
        
        intermediary_edgelist['inter_x'] = intermediary_edgelist['x_real'] + (((intermediary_edgelist['x_real']-intermediary_edgelist['x_abstract'])/nsteps)*(i))
        intermediary_edgelist['inter_y'] = intermediary_edgelist['y_real'] +(((intermediary_edgelist['y_real']-intermediary_edgelist['y_abstract'])/nsteps)*(i))
        intermediary_edgelist['inter_z'] = intermediary_edgelist['z_real'] + (((intermediary_edgelist['z_real']-intermediary_edgelist['z_abstract'])/nsteps)*(i))
        intermediary_edgelist.rename(columns={'inter_x':'x', 'inter_y':'y', 'inter_z':'z'}, inplace=True)
        interEdges.append(intermediary_edgelist)
    if output:
        try: os.makedirs(output)
        except: pass
        with open(os.path.join(output, 'animation_edgeframes.pkl'), 'wb') as f:
            pickle.dump(interEdges, f)
        
    return interEdges

def extract_real_abstract(G, edgelist, layout=nx.kamada_kawai_layout, output=False):
    """A wrapper for reimport_XYZ; gather dataframes for real, abstract, and all positions
    
    Keyword arguments:
        G -- a NetworkX graph object
        edgelist -- (dataframe) the extracted edgelist containing information about edges
        layout -- (object) the NetworkX graph layout algorithm to use (default: nx.kamada_kawai_layout)
        output -- (str or path) (optional) where to cache
    
    Returns: three dataframes -- real_edgelist, abstract_edgelist, extended_edgelist
    """
    
    extended_edgelist = reimport_newXYZ(edgelist, G, layout=layout)
    
    abstract_edgelist = extended_edgelist.copy()
    abstract_edgelist.rename(columns={'x_abstract':'x', 'y_abstract':'y', 'z_abstract':'z'}, inplace=True)
    
    real_edgelist = extended_edgelist.copy()
    real_edgelist.rename(columns={'x_real':'x', 'y_real':'y', 'z_real':'z'}, inplace=True)

    if output:
        try: os.makedirs(output)
        except: pass
        real_edgelist.to_csv(os.path.join(output, 'real_edgelist.csv'), index=False)
        abstract_edgelist.to_csv(os.path.join(output, 'abstract_edgelist.csv'), index=False)

    return real_edgelist, abstract_edgelist, extended_edgelist

def build_animation(G, edgelist, layout=nx.kamada_kawai_layout, output=False):
    """Build the plotly animation for interpolating between 2D and 3D
    
    
    Keyword arguments:
        G -- a NetworkX graph object
        edgelist -- (dataframe) the extracted edgelist containing information about edges
        layout -- (object) the NetworkX graph layout algorithm to use (default: nx.kamada_kawai_layout)
        output -- (str or path) (optional) where to cache
    
    Returns: A Plotly figure object
    """
    real_edgelist, abstract_edgelist, extended_edgelist = extract_real_abstract(G, edgelist, layout=layout, output=output)
    
    interEdges = generate_inter_edgelist(extended_edgelist, output=os.path.join(output,'animation_interedges') if output else False)
    
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Step Number: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 100, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    frames = [go.Frame(data=get_3d_traces(G, real_edgelist, nodeSize=2))]

    i=1
    for step_df in interEdges:
        frame = go.Frame(data=get_3d_traces(G, step_df, nodeSize=2), name=str(i))
        frames.append(frame)
        slider_step = {"args": [
            [i],
            {"frame": {"duration": 100, "redraw": True},
             "mode": "immediate",
             "transition": {"duration": 100}}
        ],
            "label": i,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        i+=1

    frames.append(go.Frame(data=get_3d_traces(G, abstract_edgelist, nodeSize=2)))

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1)
    )

    fig = go.Figure(data=get_3d_traces(G, real_edgelist, nodeSize=2),
        frames=frames
    )

    fig.update_layout(scene_camera=camera,
                    title="3D -> 2D Animation",
                    template='plotly_white',
                    height=800,
                    updatemenus=[{
                        "buttons": [
                            {
                                "args": [None, {"frame": {"duration": 100, "redraw": True},
                                                "fromcurrent": True, "transition": {"duration": 100,
                                                                                    "easing": "quadratic-in-out"}}],
                                "label": "Play",
                                "method": "animate"
                            },
                            {
                                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                                  "mode": "immediate",
                                                  "transition": {"duration": 0}}],
                                "label": "Pause",
                                "method": "animate"
                            }
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top"
                    }],
                    sliders=[sliders_dict]
                )

    if output:
        try: os.makedirs(output)
        except: pass
        fig.write_html(os.path.join(output, 'animation.html'))
        plt.savefig(os.path.join(output, 'animation.png'))
    fig.show()
    return fig