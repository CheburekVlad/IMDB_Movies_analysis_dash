import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import dash_table
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
Author: Sergeev Egor
Written in the last minute so some analysis could be not as consice as it should.
"""

df = pd.read_csv("../data/imdb_processed.tsv", sep="\t")
budget_summary = df["budget"].describe()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "IMDB Films Analysis"


app.layout = dbc.Container([
    html.H1("IMDB Movies Dataset Analysis"),
    html.P(["This dashboard provides an overview of ",
           html.A("the Kaggle IMDB Movies dataset", 
                  href="https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows",
                  target="_blank"),
           " including visualizations, PCA analysis, and clustering."]),
    dcc.Tabs([
        dcc.Tab(label="Preprocessing and data visualisation", children=[
            html.Div([
                html.H1("Raw data visualisation"),
                html.P("This section provides data visualisation and summary per variable"),

                dcc.Tabs([

                    dcc.Tab(label="Full Dataset", children=[
                        html.Div([
                            html.P("This tab shows the full dataset in an interactive table."),
                            html.Div(id="reactive-table")
                        ])
                    ]),

                    dcc.Tab(label="Variable Visualisation", children=[
                        html.Div([
                            html.P("Select a variable to visualize."),
                            dcc.Dropdown(
                                id="variable-dropdown",
                                options=[
                                    {"label": "Year", "value": "year"},
                                    {"label": "Runtime", "value": "runtime"},
                                    {"label": "Budget", "value": "budget"},
                                    {"label": "IMDB Score", "value": "imdb"},
                                    {"label": "Metacritic Score", "value": "metascore"},
                                ],
                                value="year",  # Default value
                                multi=False  # Single selection
                            ),
                            html.Div([
                                dcc.Graph(id="variable-visualization")  # This will display the selected plot
                            ])
                        ])
                    ])
                ])
            ])
        ]),

        dcc.Tab(label="PCA Analysis", children=[
            html.Div([
                html.H2("PCA Analysis"),
                html.P(
                    "PCA helps identify principal groups of variables and their contribution to the dataset's variability. "
                    "Below are two PCA analyses: one using all variables and another using a subset of variables."
                ),

                # Subtabs for the PCA analysis
                dcc.Tabs([
                    # First Sub-tab: PCA on specific columns
                    dcc.Tab(label="PCA on all data", children=[
                        html.Div([
                            # Graph for PCA (we will populate this with the backend)
                            dcc.Graph(id="pca-graph")
                        ])
                    ]),

                    dcc.Tab(label="PCA2 - Reduced", children=[
                        html.Div([
                            dcc.Graph(id="pca-reduced-1")
                        ])
                    ])
                ])
            ])
        ]),

        # Third Tab: Classification
        dcc.Tab(label="Classification", children=[
            html.Div([
                html.H2("Clustering Analysis"),
            ])
        ])
    ]),
    html.P("Sergeev Egor",style={
            'margin-top': 'auto'})
], fluid=True)

@app.callback(
    [Output("reactive-table", "children"),
     Output("variable-visualization", "figure"),
     Output("pca-graph", "figure"),
     Output("pca-reduced-1", "figure")],
    Input("variable-dropdown", "value")
)

def display_reactive_table(selected_variable):

    table = dash_table.DataTable(
        id="table",
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict('records'), 
        style_table={'height': '400px', 'overflowY': 'auto'}, 
        style_cell={'textAlign': 'center'}, 
        sort_action='native', 
        filter_action='native',
        page_action='native',
        page_current=0,
        page_size=20,
    )
    
    fig = update_graph(selected_variable)

    pca_fig = perform_pca_2(["year", "budget", "runtime", "imdb", "metascore"])

    pca_reduced_1 = perform_pca_3(["budget", "imdb", "votes", "runtime"])


    return table, fig, pca_fig , pca_reduced_1 

def update_graph(selected_variable):

    if selected_variable == "year":
        fig = px.histogram(df, x="year", nbins=30,
                            title="Histogram of Movies by Year",
                            category_orders={"year": sorted(df["year"].unique())})
    elif selected_variable == "runtime":
        fig = px.histogram(df, x="runtime", nbins=30,
                            title="Histogram of Movies by Runtime")
    elif selected_variable == "budget":
        fig = px.histogram(df, x="budget", nbins=100,
                            title="Histogram of Movies by Budget")
    elif selected_variable == "imdb":
        fig = px.histogram(df, x="imdb", nbins=18,
                            title="Histogram of Movies by IMDB Score")
    elif selected_variable == "metascore":
        fig = px.histogram(df, x="metascore", nbins=20,
                            title="Histogram of Movies by Metacritic Score")
    return fig

def perform_pca_3(cols):

    pca_data = df[cols]

    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    pca = PCA(n_components=3) 
    pca_result = pca.fit_transform(pca_data_scaled)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])

    fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', title="PCA: First Three Principal Components",
                        labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'})
    fig.update_traces(marker=dict(size=5, opacity=0.7)).update_layout(
                                                            width=1200, 
                                                            height=800, 
                                                            title='PCA: First Three Principal Components')

    return fig

def perform_pca_2(cols):

    pca_data = df[cols]

    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    pca = PCA(n_components=2) 
    pca_result = pca.fit_transform(pca_data_scaled)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

    fig = px.scatter_3d(pca_df, x='PC1', y='PC2', title="PCA: First Two Principal Components",
                        labels={'PC1': 'PC1', 'PC2': 'PC2'})
    

    return fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8080)
