#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import joblib
import numpy as np
from scipy.optimize import minimize
import os
from flask import Flask
from flask_compress import Compress

# Inicializar o servidor Flask
server = Flask(__name__)
Compress(server)  # Configurar o Flask-Compress no servidor Flask

# Inicializar o aplicativo Dash
app = Dash(__name__, server=server)  # Conectar o aplicativo Dash ao servidor Flask

# Carregar o modelo treinado e os dados normalizados
model_path = 'saved_model3.xgb'
data_path = 'Processed_Normalized_Data.csv'
allocation_path = 'best_allocation1000.csv'

# Verifique se os arquivos necessários estão presentes
if not os.path.exists(model_path) or not os.path.exists(data_path) or not os.path.exists(allocation_path):
    raise FileNotFoundError("Certifique-se de que os caminhos para o modelo e os dados estão corretos.")

# Carregar o modelo e os dados
model = joblib.load(model_path)
df = pd.read_csv(data_path)
best_allocation = pd.read_csv(allocation_path)

# Definir o layout do aplicativo
app.layout = html.Div([
    html.H1("Aplicativo de Alocação de Recursos"),
    dcc.Graph(id='output-graph'),
    dcc.Slider(
        id='my-slider',
        min=0,
        max=10,
        step=0.1,
        value=5
    ),
    html.Div(id='slider-output-container')
])

# Definir os callbacks para interatividade
@app.callback(
    Output('slider-output-container', 'children'),
    Input('my-slider', 'value')
)
def update_output(value):
    return f'Você selecionou: {value}'

@app.callback(
    Output('output-graph', 'figure'),
    Input('my-slider', 'value')
)
def update_graph(value):
    # Lógica para atualizar o gráfico com base no valor do slider
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[value, value + 1, value + 2])])
    return fig

# Iniciar o servidor
if __name__ == "__main__":
    app.run_server(debug=True)
