import pandas as pd
from scipy.cluster.hierarchy import average
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

data = pd.read_csv('../../data/features/prepared_train.csv')
data.drop(['target_reg'], axis=1, inplace=True)

target = 'target_class'
features = [col for col in data.columns if col != target]

X = data[features]
y = data[target]

if y.dtype == 'object' or y.dtype.name == 'category':
    y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

param_grid = {
    'iterations': [100, 150, 200],
    'depth': [2, 3, 4],
    'learning_rate': [0.01, 0.1, 0.05],
    'l2_leaf_reg': [2, 3],
    'border_count': [32, 64]
}

cat_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='MultiClass',
    custom_metric='Recall',
    verbose=0,
    random_seed=42
)

grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring='recall_macro',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_test, y_test),
    verbose=0,

)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
recall = recall_score(y_test, y_pred, average="macro")

print(f"Лучшая точность (Accuracy): {accuracy}")
print(f"F1 Macro: {f1_macro}")
print(f"F1 Weighted: {f1_weighted}")
print(f"ROC_AUC: {roc_auc}")
print(f"Recall: {recall}")

best_model.save_model('../../models/catboost_model.cbm')
print("Модель сохранена в файл catboost_model.cbm")

