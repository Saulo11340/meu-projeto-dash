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
from flask import Flask
from flask_compress import Compress

# Inicializar o servidor Flask
server = Flask(__name__)

# Inicializar o aplicativo Dash
app = Dash(__name__, server=server)
Compress(app)

# Carregar o modelo treinado e os dados normalizados
model_path = 'saved_model3.xgb'  # Certifique-se de que o caminho esteja correto
data_path = 'Processed_Normalized_Data.csv'  # Certifique-se de que o caminho esteja correto
allocation_path = 'best_allocation1000.csv'  # Certifique-se de que o caminho esteja correto

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# Obter a ordem correta das colunas do modelo
model_features = model.get_booster().feature_names

# Carregar a melhor alocação de um arquivo CSV
best_allocation_df = pd.read_csv(allocation_path)

# Converter a linha única do DataFrame em um dicionário
best_allocation = best_allocation_df.iloc[0].to_dict()

# Função para simular alocação orçamentária e calcular indicadores
def simulate_allocation(allocation):
    allocation_df = pd.DataFrame([allocation])
    
    # Garantir que todas as colunas necessárias estão presentes
    for col in model_features:
        if col not in allocation_df.columns:
            allocation_df[col] = 0
    
    allocation_df = allocation_df[model_features]
    y_pred = model.predict(allocation_df)
    pib, inflacao, gini_index = y_pred[0]
    return pib, inflacao, gini_index

# Função para calcular a diferença entre a saída do modelo e zero para cada indicador
def objective_function(x):
    # Constrói o dicionário de alocação a partir do vetor x
    allocation = {feature: x[i] for i, feature in enumerate(model_features)}
    pib, inflacao, gini_index = simulate_allocation(allocation)
    # Função objetivo é a soma dos quadrados das diferenças em relação a zero
    return pib**2 + inflacao**2 + gini_index**2

# Encontrar a alocação que resulta em indicadores médios (zero)
# Usando otimização numérica para minimizar a função objetivo
initial_guess = np.zeros(len(model_features))  # Começar com todos os valores iguais a zero
result = minimize(objective_function, initial_guess, method='BFGS')

# Extrair a alocação que minimiza a função objetivo
average_allocation = {feature: result.x[i] for i, feature in enumerate(model_features)}

# Função para determinar se um valor é considerado "igual à média"
def is_close_to_zero(value, tol=0.001):
    return abs(value) < tol

# Estilos globais para a aplicação
global_styles = {
    'fontFamily': 'Arial, sans-serif',
    'margin': '0 auto',
    'padding': '20px',
    'maxWidth': '1200px',
    'color': '#333'
}

# Definir o layout da aplicação
app.layout = html.Div([
    # Título
    html.H1("Budget Allocation Optimization", style={'textAlign': 'center', 'color': '#2c3e50'}),

    # Explicação sobre os sliders
    html.Div([
        html.P("The sliders below represent normalized budget allocations for different categories. Values range from -2 to 3, where 0 represents the historical average. Adjusting the sliders changes the allocation and the resulting indicators."),
        html.P("Indicator values are presented in terms of standard deviations from the mean. A positive value indicates an impact above the average, while a negative value indicates an impact below the average.")
    ], style={'textAlign': 'center', 'color': '#34495e', 'marginTop': '20px'}),

    # Sliders e controles
    html.Div([
        html.H2("Adjust Budget Allocation", style={'textAlign': 'center', 'color': '#34495e'}),
        html.Div([
            html.Div([
                html.Label(feature, style={'marginBottom': '5px', 'display': 'block', 'color': '#34495e'}),
                dcc.Slider(
                    id=f'slider-{feature.lower().replace(" ", "-")}',
                    min=-2, max=3, step=0.1, value=best_allocation[feature],
                    marks={i: {'label': f'{i:.1f}', 'style': {'transform': 'translate(-10px, -15px)'}} for i in np.arange(-2, 3.1, 0.5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin': '20px 0', 'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}) for feature in model_features
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Botões de controle: Cenários Alternativos
    html.Div([
        html.H3("Alternative Scenarios", style={'textAlign': 'center', 'color': '#34495e'}),
        html.Div([
            html.Button('Best Optimization Result', id='best-optimization', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
            html.Button('Average Values', id='average-values', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
            html.Button('Minimum Values', id='minimum-values', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': '#e67e22', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
            html.Button('Maximum Values', id='maximum-values', n_clicks=0, style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    ]),

    # Resultados dos Indicadores
    html.Div(id='output-container', style={
        'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': '#f9f9f9',
        'textAlign': 'center', 'marginBottom': '20px', 'color': '#2c3e50'
    }),

    # Gráfico de barras
    dcc.Graph(id='bar-chart', config={'displayModeBar': False}),

    # Texto explicativo sobre o gráfico
    html.Div([
        html.P("This chart shows the standard deviations of socioeconomic indicators relative to the historical average based on the adjusted budget allocation. Positive indicators represent impacts above the average, while negative ones indicate below average."),
        html.P("Use the sliders to adjust the allocation and observe how GDP, Inflation, and Gini Index indicators respond to changes.")
    ], style={'textAlign': 'center', 'color': '#34495e', 'marginTop': '20px'})

], style=global_styles)

# Callback para atualizar sliders e gráfico de barras com base na entrada do usuário
@app.callback(
    [Output(f'slider-{feature.lower().replace(" ", "-")}', 'value') for feature in model_features] +
    [Output('bar-chart', 'figure'), Output('output-container', 'children')],
    [Input(f'slider-{feature.lower().replace(" ", "-")}', 'value') for feature in model_features] +
    [Input('best-optimization', 'n_clicks'), Input('average-values', 'n_clicks'),
     Input('minimum-values', 'n_clicks'), Input('maximum-values', 'n_clicks')],
    [State(f'slider-{feature.lower().replace(" ", "-")}', 'value') for feature in model_features]
)
def update_output(*args):
    from dash import callback_context
    ctx = callback_context

    # Determinar os valores dos sliders a partir das entradas
    slider_values = list(args[:len(model_features)])
    best_clicks, avg_clicks, min_clicks, max_clicks = args[len(model_features):len(model_features) + 4]
    slider_values_state = args[len(model_features) + 4:]

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Reagir aos botões de cenário
    if 'best-optimization' in ctx.triggered[0]['prop_id']:
        slider_values = [best_allocation[feature] for feature in model_features]
    elif 'average-values' in ctx.triggered[0]['prop_id']:
        slider_values = [average_allocation[feature] for feature in model_features]
    elif 'minimum-values' in ctx.triggered[0]['prop_id']:
        slider_values = [-2] * len(model_features)
    elif 'maximum-values' in ctx.triggered[0]['prop_id']:
        slider_values = [3] * len(model_features)

    # Calcular indicadores baseados nos valores ajustados dos sliders
    allocation = {feature: slider_values[i] for i, feature in enumerate(model_features)}
    pib, inflacao, gini_index = simulate_allocation(allocation)

    # Criar o gráfico de barras
    fig = go.Figure(data=[
        go.Bar(name='GDP', x=['GDP'], y=[pib], marker_color='rgb(26, 118, 255)'),
        go.Bar(name='Inflation', x=['Inflation'], y=[inflacao], marker_color='rgb(55, 83, 109)'),
        go.Bar(name='Gini Index', x=['Gini Index'], y=[gini_index], marker_color='rgb(50, 171, 96)')
    ])
    fig.update_layout(barmode='group', xaxis_title='Indicators', yaxis_title='Standard Deviations from Average')

    # Atualizar o texto do output
    output_text = f"GDP: {pib:.2f}, Inflation: {inflacao:.2f}, Gini Index: {gini_index:.2f}"

    return slider_values, fig, output_text

if __name__ == '__main__':
    app.run_server(debug=True)
