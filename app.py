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

# Carregar o modelo treinado e os dados normalizados
model_path = 'saved_model3.xgb'
data_path = 'Processed_Normalized_Data.csv'
allocation_path = 'best_allocation1000.csv'

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

# Inicializar o aplicativo Dash
app = Dash(__name__)
compress = Compress()
compress.init_app(app.server)  # Aplicar o Flask-Compress ao servidor Flask subjacente

# Estilos globais para a aplicação
global_styles = {
    'fontFamily': 'Arial, sans-serif',
    'margin': '0 auto',
    'padding': '20px',
    'maxWidth': '1200px',
    'color': '#333'
}

# Layout da aplicação
app.layout = html.Div([
    html.H1("Budget Allocation Optimization", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.Div([
        html.P("The sliders below represent normalized budget allocations for different categories. Values range from -2 to 3, where 0 represents the historical average. Adjusting the sliders changes the allocation and the resulting indicators."),
        html.P("Indicator values are presented in terms of standard deviations from the mean. A positive value indicates an impact above the average, while a negative value indicates an impact below the average.")
    ], style={'textAlign': 'center', 'color': '#34495e', 'marginTop': '20px'}),
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
    html.Div([
        html.H3("Alternative Scenarios", style={'textAlign': 'center', 'color': '#34495e'}),
        html.Div([
            html.Button('Best Optimization Result', id='best-optimization', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
            html.Button('Average Values', id='average-values', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
            html.Button('Minimum Values', id='minimum-values', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': '#e67e22', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
            html.Button('Maximum Values', id='maximum-values', n_clicks=0, style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    ]),
    html.Div(id='output-container', style={
        'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': '#f9f9f9',
        'textAlign': 'center', 'marginBottom': '20px', 'color': '#2c3e50'
    }),
    dcc.Graph(id='bar-chart', config={'displayModeBar': False}),
    html.Div([
        html.P("This chart shows the standard deviations of socioeconomic indicators relative to the historical average based on the adjusted budget allocation. Positive indicators represent impacts above the average, while negative ones indicate below average."),
        html.P("Use the sliders to adjust the allocation and observe how GDP, Inflation, and Gini Index indicators respond to changes.")
    ], style={'textAlign': 'center', 'color': '#34495e', 'marginTop': '20px'})
], style=global_styles)

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

    slider_values = list(args[:len(model_features)])
    best_clicks, avg_clicks, min_clicks, max_clicks = args[len(model_features):len(model_features)+4]

    if ctx.triggered:
        if ctx.triggered[0]['prop_id'] == 'best-optimization.n_clicks':
            slider_values = [best_allocation[feature] for feature in model_features]
        elif ctx.triggered[0]['prop_id'] == 'average-values.n_clicks':
            slider_values = [average_allocation[feature] for feature in model_features]
        elif ctx.triggered[0]['prop_id'] == 'minimum-values.n_clicks':
            slider_values = [-2] * len(model_features)
        elif ctx.triggered[0]['prop_id'] == 'maximum-values.n_clicks':
            slider_values = [3] * len(model_features)

    allocation = {feature: value for feature, value in zip(model_features, slider_values)}
    pib, inflacao, gini_index = simulate_allocation(allocation)

    bar_fig = go.Figure(data=[
        go.Bar(name='GDP', x=['Current Allocation'], y=[pib], marker_color='green', text=f'Current: {pib:.2f} std devs'),
        go.Bar(name='Inflation', x=['Current Allocation'], y=[inflacao], marker_color='blue', text=f'Current: {inflacao:.2f} std devs'),
        go.Bar(name='Gini Index', x=['Current Allocation'], y=[gini_index], marker_color='purple', text=f'Current: {gini_index:.2f} std devs')
    ])
    bar_fig.update_layout(
        title="Comparison of Socioeconomic Indicators with the Average",
        xaxis_title="Category",
        yaxis_title="Standard Deviations from the Mean",
        barmode='group',
        legend_title="Indicators",
        yaxis=dict(showgrid=True, zeroline=True, showticklabels=True, range=[-3, 3]),
        plot_bgcolor='#f9f9f9'
    )

    output_text = html.Div([
        html.Div([
            html.Span("📊 GDP: "),
            html.Span(f"{abs(pib):.2f} standard deviations ", style={'fontWeight': 'bold'}),
            html.Span('above' if pib > 0 else 'equal to the average' if is_close_to_zero(pib) else 'below'),
            html.Span(" the mean.")
        ]),
        html.Div([
            html.Span("📉 Inflation: "),
            html.Span(f"{abs(inflacao):.2f} standard deviations ", style={'fontWeight': 'bold'}),
            html.Span('above' if inflacao > 0 else 'equal to the average' if is_close_to_zero(inflacao) else 'below'),
            html.Span(" the mean.")
        ]),
        html.Div([
            html.Span("💰 Gini Index: "),
            html.Span(f"{abs(gini_index):.2f} standard deviations ", style={'fontWeight': 'bold'}),
            html.Span('above' if gini_index > 0 else 'equal to the average' if is_close_to_zero(gini_index) else 'below'),
            html.Span(" the mean.")
        ])
    ])

    return (*slider_values, bar_fig, output_text)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host='0.0.0.0', port=port)
