# IMDB_Movies_analysis_dash
Studies project for movie analysis

Preprocessing was done via `src/preprocessing.R`

In order to visualize the dashboard conda is needed to ensure the correct work of all depndecies:
`apt install conda` (or check official documentation: [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Then you need to initialize the environment

`conda env create -p environment.yaml`

Last step is to launch the script:

`python3 src/dashboard.yaml`
and open the app:
`http://localhost:8080/`
