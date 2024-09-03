#!/usr/bin/env python
# coding: utf-8

# Importar bibliotecas necessárias
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import joblib
import numpy as np
from scipy.optimize import minimize
import os
from flask_compress import Compress
import flask

# Inicializar o servidor Flask
server = flask.Flask(__name__)

# Inicializar o aplicativo Dash
app = Dash(__name__, server=server)  # Vincular o Dash ao servidor Flask
compress = Compress()
compress.init_app(server)  # Aplicar o Flask-Compress ao servidor Flask subjacente

# Carregar o modelo treinado e os dados normalizados
model_path = 'saved_model3.xgb'
data_path = 'Processed_Normalized_Data.csv'
allocation_path = 'best_allocation1000.csv'

# Certifique-se de que os arquivos estão no diretório correto
if not os.path.exists(model_path) or not os.path.exists(data_path) or not os.path.exists(allocation_path):
    raise FileNotFoundError("One or more required files are missing. Please check the paths.")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# Obter a ordem correta das colunas do modelo
model_features = model.get_booster().feature_names

# Carregar a melhor alocação de um arquivo CSV
best_allocation_df = pd.read_csv(allocation_path)
best_allocation = best_allocation_df.iloc[0].to_dict()

# Funções de alocação e otimização
def simulate_allocation(allocation):
    allocation_df = pd.DataFrame([allocation])
    for col in model_features:
        if col not in allocation_df.columns:
            allocation_df[col] = 0
    allocation_df = allocation_df[model_features]
    y_pred = model.predict(allocation_df)
    pib, inflacao, gini_index = y_pred[0]
    return pib, inflacao, gini_index

def objective_function(x):
    allocation = {feature: x[i] for i, feature in enumerate(model_features)}
    pib, inflacao, gini_index = simulate_allocation(allocation)
    return pib**2 + inflacao**2 + gini_index**2

initial_guess = np.zeros(len(model_features))
result = minimize(objective_function, initial_guess, method='BFGS')
average_allocation = {feature: result.x[i] for i, feature in enumerate(model_features)}

def is_close_to_zero(value, tol=0.001):
    return abs(value) < tol

# Layout e callbacks do Dash aqui...

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host='0.0.0.0', port=port)
