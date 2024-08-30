Meu Projeto Dash

Este repositório contém uma aplicação desenvolvida em Dash para otimização de recursos públicos orçamentários. A aplicação utiliza modelos preditivos para simular diferentes alocações orçamentárias e analisar o impacto dessas alocações em indicadores socioeconômicos, como PIB, inflação e índice de Gini.

Conteúdo do Repositório

app.py: Código principal da aplicação Dash, incluindo a interface de usuário e a lógica para simulação de alocação orçamentária.

requirements.txt: Lista de dependências necessárias para executar a aplicação. Inclui bibliotecas como Dash, Plotly, Pandas, NumPy e SciPy.

Processed_Normalized_Data.csv: Conjunto de dados normalizados usados pela aplicação para as previsões do modelo.

best_allocation1000.csv: Arquivo CSV contendo a melhor alocação orçamentária calculada pelo modelo.

saved_model3.xgb: Modelo treinado utilizando XGBoost, usado pela aplicação para prever os resultados de diferentes alocações orçamentárias.

Procfile.txt: Arquivo de configuração para implantação em ambientes como o Render, especificando como iniciar a aplicação.
Instalação e Execução
Para rodar a aplicação localmente, siga os passos abaixo:

Clone o repositório:

bash
Copiar código
git clone https://github.com/Saulo11340/meu-projeto-dash.git
cd meu-projeto-dash
Crie um ambiente virtual e ative-o:

bash
Copiar código
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
Instale as dependências:

bash
Copiar código
pip install -r requirements.txt
Execute a aplicação:

bash
Copiar código
python app.py
Acesse a aplicação no seu navegador:
A aplicação estará disponível em http://127.0.0.1:8050.

Implantação
A aplicação pode ser implantada em plataformas como o Render. O arquivo Procfile.txt está configurado para iniciar a aplicação automaticamente. Certifique-se de que todas as dependências estejam listadas no requirements.txt e que os arquivos de dados necessários estejam presentes no ambiente de implantação.

Como Funciona
Importação de Dados e Modelos: A aplicação carrega os dados normalizados e o modelo preditivo treinado.
Simulação de Alocação Orçamentária: O usuário pode ajustar sliders para simular diferentes alocações orçamentárias e visualizar os efeitos nos indicadores socioeconômicos.
Otimização: A aplicação utiliza otimização numérica para encontrar alocações que minimizam as diferenças em relação aos indicadores de referência.
Visualização: Resultados são apresentados em um gráfico de bolhas interativo e através de uma interface de usuário intuitiva.
