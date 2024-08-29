#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Instalação das bibliotecas necessárias
get_ipython().system('pip install pandas plotly dash joblib numpy scipy --quiet')

# Importar bibliotecas necessárias
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import joblib
import numpy as np
from scipy.optimize import minimize

# Carregar o modelo treinado e os dados normalizados
model = joblib.load('saved_model3.pkl')
df = pd.read_csv('Processed_Normalized_Data.csv')

# Obter a ordem correta das colunas do modelo
model_features = model.get_booster().feature_names

# Carregar a melhor alocação de um arquivo CSV
best_allocation_df = pd.read_csv('best_allocation1000.csv')

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

# Inicializar o aplicativo Dash
app = Dash(__name__)

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
    # Título
    html.H1("Otimização de Recursos Públicos Orçamentários", style={'textAlign': 'center', 'color': '#2c3e50'}),

    # Sliders e controles
    html.Div([
        html.H2("Ajuste de Alocação Orçamentária", style={'textAlign': 'center', 'color': '#34495e'}),
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

    # Botões de controle
    html.Div([
        html.Button('Aplicar Melhor Otimização', id='apply-optimization', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
        html.Button('Restaurar Valores Médios', id='restore-average', n_clicks=0, style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Resultados dos Indicadores
    html.Div(id='output-container', style={
        'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': '#f9f9f9',
        'textAlign': 'center', 'marginBottom': '20px', 'color': '#2c3e50'
    }),

    # Gráfico de bolhas
    dcc.Graph(id='bubble-chart'),

    # Legenda
    html.Div([
        html.H3("Legenda", style={'textAlign': 'center', 'color': '#34495e'}),
        html.Div([
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                html.Div(style={'width': '20px', 'height': '20px', 'backgroundColor': 'green', 'borderRadius': '50%', 'marginRight': '10px'}),
                html.P("PIB positivo", style={'margin': 0})
            ]),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                html.Div(style={'width': '20px', 'height': '20px', 'backgroundColor': 'red', 'borderRadius': '50%', 'marginRight': '10px'}),
                html.P("PIB negativo", style={'margin': 0})
            ]),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                html.Div(style={'width': '20px', 'height': '20px', 'backgroundColor': 'blue', 'borderRadius': '50%', 'marginRight': '10px'}),
                html.P("Inflação negativa (boa)", style={'margin': 0})
            ]),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                html.Div(style={'width': '20px', 'height': '20px', 'backgroundColor': 'orange', 'borderRadius': '50%', 'marginRight': '10px'}),
                html.P("Inflação positiva (ruim)", style={'margin': 0})
            ]),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                html.Div(style={'width': '20px', 'height': '20px', 'backgroundColor': 'purple', 'borderRadius': '50%', 'marginRight': '10px'}),
                html.P("Redução do Índice Gini (boa)", style={'margin': 0})
            ]),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                html.Div(style={'width': '20px', 'height': '20px', 'backgroundColor': 'yellow', 'borderRadius': '50%', 'marginRight': '10px'}),
                html.P("Aumento do Índice Gini (ruim)", style={'margin': 0})
            ]),
        ], style={'textAlign': 'left', 'display': 'inline-block', 'marginLeft': '20px'})
    ], style={'marginTop': '20px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': '#f1f1f1'})
], style=global_styles)

# Callback para atualizar sliders e gráfico de bolhas com base na entrada do usuário
@app.callback(
    [Output(f'slider-{feature.lower().replace(" ", "-")}', 'value') for feature in model_features] +
    [Output('bubble-chart', 'figure'), Output('output-container', 'children')],
    [Input(f'slider-{feature.lower().replace(" ", "-")}', 'value') for feature in model_features] +
    [Input('apply-optimization', 'n_clicks'), Input('restore-average', 'n_clicks')],
    [State(f'slider-{feature.lower().replace(" ", "-")}', 'value') for feature in model_features]
)
def update_output(*args):
    from dash import callback_context
    ctx = callback_context

    # Determinar os valores dos sliders a partir das entradas
    slider_values = list(args[:len(model_features)])
    apply_clicks, restore_average_clicks = args[len(model_features):len(model_features)+2]

    # Verificar qual botão foi pressionado ou se algum slider foi alterado
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'apply-optimization.n_clicks':
        # Atualizar sliders com a melhor otimização carregada do CSV
        slider_values = [best_allocation[feature] for feature in model_features]
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'restore-average.n_clicks':
        # Restaurar sliders para os valores médios (que geram resultados zero)
        slider_values = [average_allocation[feature] for feature in model_features]

    # Calcular alocação simulada com base nos valores dos sliders
    allocation = {feature: value for feature, value in zip(model_features, slider_values)}
    pib, inflacao, gini_index = simulate_allocation(allocation)

    # Ajustar tamanho e cor das bolhas com base nos resultados
    size_values = [
        max(10, pib * 100),
        max(10, 100 - abs(inflacao * 100)),
        max(10, 100 - abs(gini_index * 100))
    ]
    colors = ['green' if pib > 0 else 'red',
              'blue' if inflacao < 0 else 'orange',
              'purple' if gini_index < 0 else 'yellow']
    labels = ['PIB', 'Inflação', 'Índice Gini']

    # Criar gráfico de bolhas
    fig = go.Figure()
    for i in range(3):
        fig.add_trace(go.Scatter(
            x=[labels[i]],
            y=[0],
            mode='markers',
            marker=dict(size=size_values[i], color=colors[i], opacity=0.8),
            text=f"{labels[i]}: {size_values[i]:.2f}",
            hoverinfo="text"
        ))

    fig.update_layout(
        title="Impacto da Alocação Orçamentária nos Indicadores Socioeconômicos",
        xaxis_title="Indicadores",
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False),
        showlegend=False
    )

    # Ajuste no texto descritivo para refletir se é acima, abaixo ou igual à média
    output_text = html.Div([
        html.Div([
            html.Span("📊 PIB: "),
            html.Span(f"{abs(pib):.2f} desvios padrão ", style={'fontWeight': 'bold'}),
            html.Span('acima' if pib > 0 else 'igual à média' if is_close_to_zero(pib) else 'abaixo'),
            html.Span(" da média.")
        ]),
        html.Div([
            html.Span("📉 Inflação: "),
            html.Span(f"{abs(inflacao):.2f} desvios padrão ", style={'fontWeight': 'bold'}),
            html.Span('acima' if inflacao > 0 else 'igual à média' if is_close_to_zero(inflacao) else 'abaixo'),
            html.Span(" da média.")
        ]),
        html.Div([
            html.Span("💰 Índice Gini: "),  # Alterado para um ícone de dinheiro para representar distribuição de renda
            html.Span(f"{abs(gini_index):.2f} desvios padrão ", style={'fontWeight': 'bold'}),
            html.Span('acima' if gini_index > 0 else 'igual à média' if is_close_to_zero(gini_index) else 'abaixo'),
            html.Span(" da média.")
        ])
    ])

    return (*slider_values, fig, output_text)

# Executar o servidor dentro do notebook
app.run_server(mode='inline', debug=True)


# In[ ]:




