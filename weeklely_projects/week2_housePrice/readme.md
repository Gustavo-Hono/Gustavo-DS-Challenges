# 🏠 Previsão de Preços de Imóveis (Kaggle)

Este repositório contém minha solução para a competição do Kaggle **[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/home-data-for-ml-course)**. O objetivo é prever o preço final de venda de casas em Ames, Iowa, com base em 79 features diferentes.

Este projeto foi desenvolvido como parte do curso "Intro to Machine Learning" do Kaggle, cobrindo todo o pipeline de um projeto de ciência de dados, desde a exploração inicial até o envio de previsões.

## ⚙️ Fluxo de Trabalho do Projeto

A análise foi realizada em um Jupyter Notebook (`main.ipynb`) e seguiu as seguintes etapas:

1.  **Carregamento dos Dados:** Leitura dos arquivos `train.csv` e `test.csv`.
2.  **Análise Exploratória e Limpeza:**
    * Identificação da variável alvo (`SalePrice`).
    * Tratamento de dados ausentes (`NaN`), por exemplo, preenchendo o `LotFrontage` com o valor médio da coluna.
3.  **Engenharia de Features (Pré-processamento):**
    * Identificação de colunas categóricas (ambas do tipo `object` e numéricas que representam categorias, como `MSSubClass`).
    * Conversão de todas as colunas categóricas para um formato numérico usando **One-Hot Encoding** (`pd.get_dummies()`).
4.  **Alinhamento das Features:**
    * Um passo crucial foi garantir que os dados de treino e teste tivessem exatamente as mesmas colunas após o *encoding*. Isso foi feito usando `.reindex()`, o que evitou erros de `feature mismatch` durante a predição.
5.  **Divisão dos Dados:** Separação dos dados de treino em conjuntos de `treino` (80%) e `validação` (20%) para avaliar os modelos localmente.

## 🚀 Modelagem e Resultados

Dois modelos de regressão baseados em árvores foram treinados e comparados. A métrica de avaliação utilizada foi o **Mean Absolute Error (MAE)**, que representa o erro médio, em dólares, das previsões.

| Modelo | MAE no Kaggle (Public Score) |
| :--- | :--- |
| **Random Forest Regressor** | `$23.365` |
| **XGBoost Regressor (Otimizado)** | **`$22.335`** |

### Conclusões:

* O modelo **Random Forest** serviu como uma ótima *baseline* inicial.
* A migração para o **XGBoost** (Extreme Gradient Boosting), utilizando a função de `early_stopping` para encontrar o número ideal de árvores, resultou em uma **redução de mais de $1.000** no erro médio, demonstrando ser um algoritmo mais poderoso para este conjunto de dados.

## 🛠️ Tecnologias Utilizadas

* Python 3
* Pandas
* Numpy
* Scikit-learn (para `RandomForestRegressor`, `train_test_split`, `mean_absolute_error`)
* XGBoost (para `XGBRegressor`)
* Jupyter Notebook