import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print(type(train))

except:
    print("Não deu certo")
    

print("\nVISUALIZAÇÃO DO DATASET\n")

    
def info(df):
        print(df.info())
        return df
        
def preprocess(df):
        print(df.isnull().sum())
        return df
        
def describe(df):
        print(df.describe())
        return df
        
def delete_column(df, target_column):
        df = df.drop(columns=[target_column])
        print(df.head())
        return df
        
preprocess(train)
describe(train)
train = delete_column(train, "Cabin")

women = train.loc[train.Sex == 'female']['Survived']
rate_woman = sum(women)/ len(women)

men = train.loc[train.Sex == "male"]["Survived"]
rate_men = sum(men)/len(men)

model = RandomForestClassifier(class_weight="balanced", random_state=40, n_estimators=200)



features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
y = train["Survived"]
x = pd.get_dummies(train[features])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc:.4f}")

X_final = pd.get_dummies(test[features])
predictions = model.predict(X_final)


# output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
# output.to_csv('submission.csv', index=False)
# print("Your submission was successfully saved!")

import matplotlib.pyplot as plt

print(model)

importances = model.feature_importances_
feature_names = X_train.columns

# Criar dataframe para visualização
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

print("Importância das Features:")
print(fi_df)

# Grafico
plt.figure(figsize=(6,4))
plt.barh(fi_df['Feature'], fi_df['Importance'])
plt.gca().invert_yaxis()
plt.title("RandomForest - Feature Importance")
plt.show()

