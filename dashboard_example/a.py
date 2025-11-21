import dash                              # pip install dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input

from dash_extensions import Lottie       # pip install dash-extensions
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import plotly.express as px              # pip install plotly
import pandas as pd                      # pip install pandas
from datetime import date
import calendar
from wordcloud import WordCloud          # pip install wordcloud


# Bootstrap themes: https://hellodash.pythonanywhere.com/theme_explorer
# Create a Dash object
# Flask: app = Flask(__name__)
# __name__ (default setting)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# if Jupyter Notebook/Colab
# app = JupyterDash(__name__, external_stylesheets=[dbc.themes.LUX])

"""
dbc stands for Dash Bootstrap Components
 
- For professional, responsive dashboards.
- If you want cleaner, reusable, and maintainable code.
- When using Bootstrap themes for styling.
"""



"""
mb-2 = margin-bottom: spacing * 2
mt-2 = margin-top: spacing * 2
l / s	left / start
r / e	right / end
x	left + right
y	top + bottom
"""

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardImg(src='/assets/linkedin.png')
            ],className='mb-2'), 
            # margin bottom 2 space,  0.5rem (8px) margin
            # link: https://hackerthemes.com/bootstrap-cheatsheet/
            # dbc.card: https://dash-bootstrap-components.opensource.faculty.ai/docs/components/card/
            dbc.Card([  
                dbc.CardBody([
                ])
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=8),
    ],className='mb-2 mt-2'), # or: my-2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=2),
    ],className='mb-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=4),
    ],className='mb-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                ])
            ]),
        ], width=4),
    ],className='mb-2'),
], fluid=True)



# It is used to ensure that certain parts of the code are executed 
# only when the script is run directly, and not when it is imported as a module in another script.

if __name__=='__main__':
# you can change the port number here
    app.run_server(debug=False, port=8001)  # Starts the server only if run directly.
    
    
# if jupyter notebook/colab
# app.run_server(mode='jupyterlab', port=8001)