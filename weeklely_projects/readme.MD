![kaggle](https://img.shields.io/badge/Kaggle-Score%200.77990-blue)
![status](https://img.shields.io/badge/Status-Em%20Progresso-yellow)
![python](https://img.shields.io/badge/Python-3.10-brightgreen)


# 🛳️ Titanic Survival Prediction – Weekly ML Project

Este projeto faz parte de uma rotina semanal de prática em Machine Learning, com o objetivo de desenvolver habilidades de análise, pré-processamento, modelagem e avaliação de dados reais.
Para iniciar esta jornada, utilizei o clássico desafio **Titanic: Machine Learning from Disaster** do Kaggle.

Acompanhe em 
https://www.youtube.com/@rumo_ao_est%C3%A1gio

---

## 📌 Objetivo

Treinar um modelo supervisionado capaz de prever quais passageiros sobreviveram ao desastre do Titanic, utilizando variáveis relacionadas ao perfil social e características individuais.

---

## 🧠 Metodologia

O fluxo implementado segue as principais etapas de um pipeline de Data Science:

1. **Carregamento dos dados**  
   `train.csv` e `test.csv` da competição do Kaggle.

2. **Análise exploratória inicial**
   - Estatísticas descritivas
   - Inspeção de valores ausentes
   - Remoção da coluna `Cabin` devido ao grande número de nulos

3. **Feature Engineering**
   - Codificação das variáveis categóricas com `get_dummies()`
   - Seleção das features:
     - `Pclass`, `Sex`, `SibSp`, `Parch`, `Fare`

4. **Divisão em treino e teste**
   - `train_test_split(test_size=0.2, random_state=42)`

5. **Treinamento do modelo**
   - Algoritmo: **RandomForestClassifier**
   - `class_weight="balanced"`
   - `n_estimators=200`
   (se aumentar o n_estimators a tendencia é ficar mais preciso)

6. **Avaliação do modelo**
   - Métrica: Acurácia
   - Acurácia obtida: **~0.79**

7. **Cálculo da Importância das Features**
   - Identificação das variáveis com maior impacto nas previsões
   - Visualização gráfica com `matplotlib`

---

## 📊 Resultados

| Feature | Importância |
|--------|-------------|
| Fare | Alta |
| Sex (male/female) | Alta |
| Pclass | Moderada |
| SibSp | Baixa |
| Parch | Baixa |

📌 Conclusão: Fatores econômicos e gênero influenciaram fortemente a sobrevivência.

---

## 🧩 Dependências

```bash
pip install pandas scikit-learn matplotlib joblib
```


## Observações
No Algoritmo: **RandomForestClassifier**, se vc deixar sem os parametros ali e usar por default, ele pega 78, ou seja os parametros ali aumentaram 1%
