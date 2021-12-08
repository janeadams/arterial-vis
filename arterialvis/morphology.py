from os import walk
from dotenv import dotenv_values
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import networkx as nx
import traceback
from sklearn.preprocessing import normalize
import numpy as np
import pickle

def get_files(path=None):
    if (path==None):
        config = dotenv_values(".env")
        path = config['SWC_SAVE']
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files

def filter_files(files, colored=True):
    filtered_files = []
    for filename in files:
        if colored and ("ColorCoded" in filename):
                filtered_files.append(filename)
        if not colored and ("ColorCoded" not in filename):
                filtered_files.append(filename)
    return filtered_files

def get_edgelist(file, path=None, output=False):
    if output:
        try:
            edgelist = pd.read_csv(output+'.csv')
            return edgelist
        except:
            pass
    if (path==None):
        config = dotenv_values(".env")
        path = config['SWC_SAVE']
    edgelist = pd.read_csv(path+'/'+file,
                     delim_whitespace=True,
                     header=None,
                     skiprows=1).rename(columns={0:'source',
                                                 1:'color',
                                                 6:'target',
                                                 2:'x',
                                                 3:'y',
                                                 4:'z',
                                                 5:'length'})
    if output: edgelist.to_csv(output+'.csv', index=False)
    return edgelist

def generate_graph(edgelist, output=False):
    if output:
        try:
            G = pd.read_pickle(output+'.pkl')
            print(f'Found saved file at {output}.pkl')
            return G
        except:
            pass
    G=nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr='length')
    nx.set_node_attributes(G, pd.Series(edgelist['color'], index=edgelist['source']).to_dict(), 'group')
    if output: pickle.dump(G, open(output+'.pkl', "wb"))
    return G

def convert_to_edgelist(graph):
    return nx.to_pandas_edgelist(graph)

def simplifyGraph(G, output=False):
# Loop over the graph until all nodes of degree 2 have been removed and their incident edges fused
# Source: https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges
# Thank you mjkvaak, louis_guitton, and sauce_interstellaire
    
    if output:
        try:
            g = pd.read_pickle(output+'.pkl')
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
    if output: pickle.dump(g, open(output+'.pkl', "wb"))
    return g

def get_2d_positions(G, edgelist, layout=nx.kamada_kawai_layout, output=False):
    
    if output:
        try:
            edges = pd.read_pickle(output+'_edges.pkl')
            print(f'Found saved file at {output}_edges.pkl')
            nodes = pd.read_pickle(output+'_nodes.pkl')
            print(f'Found saved file at {output}_nodes.pkl')
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
            node_groups.append(edgelist.loc[edgelist['source']==node]['color'].item())

    # Bundle it all up in a dict:
    nodes = dict(
        x=node_x,
        y=node_y,
        z=[0]*len(node_x),
        name=node_name,
        groups=node_groups
    )
    if output:
        pickle.dump(edges, open(output+'_edges.pkl', "wb"))
        pickle.dump(nodes, open(output+'_nodes.pkl', "wb"))
    return edges, nodes

def get_2d_traces(G, edgelist, nodeColor=None, edgeColor=None, nodesize=1, layout=nx.kamada_kawai_layout):
    
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

            group = edgelist.loc[edgelist['source']==edge[0]]['color'].item()
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)
            edge_group.append(group)

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
        fig.write_html(output+'.html')
    return fig

def build_compound_graph(file, path=None, output=False):
    if (path==None):
        config = dotenv_values(".env")
        path = config['SWC_SAVE']
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
        fig.write_html(output+'.html')
    return fig

def build_grouped_graph(file, path=None, output=False):
    if (path==None):
        config = dotenv_values(".env")
        path = config['SWC_SAVE']
    edgelist = get_edgelist(file, path)
    g = generate_graph(edgelist)
    sparse = simplifyGraph(g)
    traces = get_3d_traces(sparse, edgelist, edgeColor='gray', nodeSize=5)
    fig=draw_graph(traces, title=file)
    if output:
        fig.write_html(output+'.html')
    return fig

def build_comparison_dashboard(path=None):
    if (path==None):
        config = dotenv_values(".env")
        path = config['SWC_SAVE']
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
    norm1 = series / np.linalg.norm(series)
    norm2 = normalize(series[:,np.newaxis], axis=0).ravel()
    return norm2

def reimport_newXYZ(edgelist, G, layout=nx.kamada_kawai_layout, z=0.5, output=False):
    if output:
        try:
            edgelist = pd.read_csv(output+'.csv')
            return edgelist
        except:
            pass
    new_edges, new_nodes = get_2d_positions(G, edgelist, layout=layout)
    newnodeDF = pd.DataFrame(new_nodes)
    newnodeDF['z']=z
    edgelist = edgelist.merge(newnodeDF, left_on='source', right_on='name', suffixes=('_real', '_abstract'))
    if output: edgelist.to_csv(output+'.csv', index=False)
    return edgelist

def generate_inter_edgelist(sourceEdgelist, nsteps=50, output=False):
    if output:
        try:
            interEdges = pd.read_pickle(output+'.pkl')
            print(f'Found saved file at {output}.pkl')
            return interEdges
        except:
            pass
    interEdges = []
    for i in np.arange(1,nsteps):
        intermediary_edgelist = sourceEdgelist.copy()
        intermediary_edgelist['inter_x'] = intermediary_edgelist['x_real'] + (((intermediary_edgelist['x_abstract']-intermediary_edgelist['x_real'])/nsteps)*i)
        intermediary_edgelist['inter_y'] = intermediary_edgelist['y_real'] + (((intermediary_edgelist['y_abstract']-intermediary_edgelist['y_real'])/nsteps)*i)
        intermediary_edgelist['inter_z'] = intermediary_edgelist['z_real'] + (((intermediary_edgelist['z_abstract']-intermediary_edgelist['z_real'])/nsteps)*i)
        intermediary_edgelist.rename(columns={'inter_x':'x', 'inter_y':'y', 'inter_z':'z'}, inplace=True)
        interEdges.append(intermediary_edgelist)
    if output: pickle.dump(interEdges, open(output+'.pkl', "wb"))
    return interEdges

def build_animation(file=None, G=None, edgelist=None, real_edgelist=None, abstract_edgelist=None, interEdges=None, layout=nx.kamada_kawai_layout, output=False):
    
    if (G==None) and (edgelist==None):
        if file is None:
            raise FileNotFoundError('You must supply either a graph object, edgelist, or filename!')
        else:
            edgelist = get_edgelist(file,output=(output+'/animation_original_edgelist' if output else False))
            G = generate_graph(edgelist,output=(output+'/animation_graph' if output else False))
        
    edgelist = convert_to_edgelist(G)
    newedgelist = reimport_newXYZ(edgelist, G, layout=layout, output=(output+'/animation_full_edgelist' if output else False))
    
    if (real_edgelist==None) or (abstract_edgelist==None):
        abstract_edgelist = newedgelist.copy()
        abstract_edgelist.rename(columns={'x_abstract':'x', 'y_abstract':'y', 'z_abstract':'z'}, inplace=True)
        real_edgelist = newedgelist.copy()
        real_edgelist.rename(columns={'x_real':'x', 'y_real':'y', 'z_real':'z'}, inplace=True)
    
    if interEdges==None:
        interEdges = generate_inter_edgelist(newedgelist, nsteps=5, output=(output+'/animation_interedges' if output else False))
    
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
        "transition": {"duration": 300, "easing": "cubic-in-out"},
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
            {"frame": {"duration": 300, "redraw": True},
             "mode": "immediate",
             "transition": {"duration": 300}}
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
                                "args": [None, {"frame": {"duration": 500, "redraw": True},
                                                "fromcurrent": True, "transition": {"duration": 300,
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

    fig.show()
    return fig