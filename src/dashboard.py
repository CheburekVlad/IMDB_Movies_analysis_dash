import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import dash_table
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from math import log10

"""
Author: Sergeev Egor
Written in a hurry so some analysis could be not as consice as it should.
"""

df = pd.read_csv("data/imdb_processed.tsv", sep="\t")
gross_summary = df["gross"].describe()


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
                html.H2("Raw data visualisation"),
                dcc.Markdown("This section provides data visualisation and summary per variable"),

                dcc.Tabs([

                    dcc.Tab(label="Full Dataset", children=[
                        html.Div([
                            html.P("Feel free to explore the dataset below"),
                            html.Div(id="reactive-table")
                        ])
                    ]),

                    dcc.Tab(label="Per variable visualisation", children=[
                        html.Div([
                            html.P("Please select a variable:"),
                            dcc.Dropdown(
                                id="variable-dropdown",
                                options=[
                                    {"label": "Year", "value": "year"},
                                    {"label": "Runtime", "value": "runtime"},
                                    {"label": "gross", "value": "gross"},
                                    {"label": "IMDB Score", "value": "imdb"},
                                    {"label": "Meta Score", "value": "metascore"},
                                ],
                                value="year",  
                                multi=False 
                            ),
                            html.Div([
                                dcc.Graph(id="variable-visualization")
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
                    "Below are two PCA analyses: one using all variables and another using a subset."
                ),

                dcc.Tabs([
                    dcc.Tab(label="PCA on all data", children=[
                        dcc.Markdown("All variables create a chaotic spread"),
                        html.Div([
                            dcc.Graph(id="pca-graph")
                        ]),
                        html.Div([
                            dcc.Graph(id="pca-hist")
                        ])
                    ]),

                    dcc.Tab(label="PCA2 - Reduced", children=[
                        dcc.Markdown("While reducing the size of the variables keeping only Gross, Number of votes, title_length and runtime gives a more concise result"),
                        html.Div([
                            dcc.Graph(id="pca-reduced-1")
                        ]),
                        html.Div([
                            dcc.Graph(id="pca-reduced-hist")
                        ])
                    ]),
                    
                ])
            ])
        ]),

        dcc.Tab(label="Classification", children=[
            html.Div([
                html.H2("Clustering Analysis"),
                dcc.Markdown("Below are clustering analyses using KMeans and DBSCAN."),

                dcc.Tabs([
                    dcc.Tab(label="KMeans Clustering", children=[
                        html.Div([
                            dcc.Markdown("Classification of the data in K = 4 clusters shows the following result:"),
                            dcc.Graph(id="kmeans-graph") 
                        ])
                    ]),

                    dcc.Tab(label="DBSCAN Clustering", children=[
                        html.Div([
                            html.P("With the values of eps=0.6 and min_samples=5 these are DBSCAN clustering results:"),
                            dcc.Graph(id="dbscan-graph")  
                        ])
                    ])
                ])
            ])
        ]),
        dcc.Tab(label="Result", children=[
            dcc.Markdown("""
                Analysis of the top 1000 IMDB films filtered to 750 has enabled me to identify four classes of film based on user votes, gross revenue and running time:\n

                1. **Non-popular films**: very low number of votes, very low gross receipts, generally short, low-budget, niche films or amateur films that seems to be easy-to-make.\n
                2. **Popular films**: lots of votes, high gross receipts, basically 2h-2h30 of screentime, very popular blockbusters.\n
                3. **Long, unpopular films**: high volume of Hours (2h30+), leads to average gross receipts, long but unpopular films.\n
                4. **Films in the middle of the film revenue scale**: slightly above average, not very “long”, not very “popular”.\n

                In the analysis DBSCAN shows rather poorer results than K -means. Probably due to the many outliers present, which it was not able to deal with to any great extent. \n
            """)
        ])
    ]),
    html.P("Sergeev Egor",style={
            'margin-top': 'auto'})
], fluid=True)

@app.callback(
    [Output("reactive-table", "children"),
     Output("variable-visualization", "figure"),
     Output("pca-graph", "figure"),
     Output("pca-hist", "figure"),
     Output("pca-reduced-1", "figure"),
     Output("pca-reduced-hist", "figure"),
     Output("kmeans-graph", "figure"),
     Output("dbscan-graph", "figure")],
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

    pca_fig, hist = perform_pca(["year", "gross", "runtime", "imdb", "metascore", "title_length", "genre_num", "votes"])

    pca_reduced_1 , hist_reduced = perform_pca(["gross", "votes", "title_length", "runtime"])

    kmeans_fig = perform_kmeans_clustering(["votes", "gross", "runtime"])


    dbscan_fig = perform_dbscan_clustering(["votes", "gross", "runtime"])

    return table, fig, pca_fig, hist, pca_reduced_1, hist_reduced , kmeans_fig, dbscan_fig

def update_graph(selected_variable):

    if selected_variable == "year":
        fig = px.histogram(df, x="year", nbins=30,
                            title="Histogram of Movies by Year",
                            category_orders={"year": sorted(df["year"].unique())})
    elif selected_variable == "runtime":
        fig = px.histogram(df, x="runtime", nbins=30,
                            title="Histogram of Movies by Runtime")
    elif selected_variable == "gross":
        fig = px.histogram(df, x="gross", nbins=100,
                            title="Histogram of Movies by Income")
    elif selected_variable == "imdb":
        fig = px.histogram(df, x="imdb", nbins=18,
                            title="Histogram of Movies by IMDB Score")
    elif selected_variable == "metascore":
        fig = px.histogram(df, x="metascore", nbins=20,
                            title="Histogram of Movies by Metacritic Score")
    return fig

def perform_pca(cols):

    pca_data = df[cols]
    
    pca_data = pca_data.dropna()
    
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(pca_data_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    if len(explained_variance) < 4:  
        explained_variance = list(explained_variance) + [0] * (4 - len(explained_variance))
    
    explained_variance_df = pd.DataFrame({
        'Principal Component': [f"PC{i+1}" for i in range(len(explained_variance))],
        'Explained Variance': explained_variance
    })


    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    scatter_fig = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3',
        title="PCA: First Three Principal Components",
        labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'}
    )
    scatter_fig.update_traces(marker=dict(size=5, opacity=0.7)).update_layout(width=1200, height=800)

    histogram_fig = px.bar(
        explained_variance_df,
        x='Principal Component',
        y='Explained Variance',
        title="Eigen Values of the First 4 Principal Components",
        labels={'Explained Variance': 'Eigen Values'},
    )
    histogram_fig.update_layout(width=800, height=500)

    return scatter_fig, histogram_fig

def perform_kmeans_clustering(columns):
    clustering_data = df[columns]
    if "gross" in clustering_data:
        clustering_data['gross'] = np.log10(pd.to_numeric(clustering_data['gross']))
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    kmeans = KMeans(n_clusters=4)
    clustering_data['Cluster'] = kmeans.fit_predict(clustering_data_scaled)
    fig = px.scatter_3d(clustering_data, x=columns[0], y=columns[1], z=columns[2], color='Cluster', 
                        title="KMeans Clustering",
                        labels={columns[0]: columns[0], columns[1]: columns[1], columns[2]: columns[2]})
    fig.update_traces(marker=dict(size=5, opacity=0.7)).update_layout(width=1200, height=800)
    return fig


def perform_dbscan_clustering(columns):

    clustering_data = df[columns]
    if "gross" in clustering_data:
        clustering_data['gross'] = np.log10(pd.to_numeric(clustering_data['gross']))

    clustering_data = clustering_data.dropna()
    
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    
    # Estimated with kNN via dbscan library in R
    dbscan = DBSCAN(eps=0.6, min_samples=5)
    clustering_data['Cluster'] = dbscan.fit_predict(clustering_data_scaled)
    

    fig = px.scatter_3d(clustering_data, x=columns[0], y=columns[1], z=columns[2], color='Cluster', 
                        title="DBSCAN Clustering",
                        labels={columns[0]: columns[0], columns[1]: columns[1], columns[2]: columns[2]})
    fig.update_traces(marker=dict(size=5, opacity=0.7)).update_layout(width=1200, height=800)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8080)
