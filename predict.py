import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import time
import datetime
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# Создание DataFrame
features = pd.read_csv('features.csv')
features_test = pd.read_csv('features_test.csv')
#Целевая переменная
y = features['radiant_win']

#Исключение признаков, которые не относятся к первым пяти минутам матча
features = features.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'], axis=1)

#Определение столбцов, в которых есть пропущенные значения
l = len(features)
count = features.count()
for i in range(0, len(count)):
    if count[i] != l:

        print(count.index[i])

#Заполнение пустых ячеек на 0
X = features.fillna(0)
features_test = features_test.fillna(0)

#Создание 5 блоков
cv = KFold(n_splits=5, shuffle=True)
cv.get_n_splits(X)

#Определение оптимального количества деревьев

# numbers = [10, 20, 30, 40, 50, 100]
# for n in numbers:
#     print(n)
#     gradient = GradientBoostingClassifier(n_estimators=n, random_state=241)
#     start_time = datetime.datetime.now()
#
#     scores = cross_val_score(gradient, X, y, cv=cv, scoring='roc_auc')
#
#     print('Time elapsed:', datetime.datetime.now() - start_time)
#     print(np.mean(scores))

#Функция для определения оптимального параметра регуляризации
def logreg (X, y):
    scores_log = np.zeros((0,1))
    for C in range (-5, 5):
        model = LogisticRegression(penalty='l2', C=10**C, random_state=241)
        start_time = datetime.datetime.now()

        kach = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

        print('Time elapsed:', datetime.datetime.now() - start_time)
        print('C=', 10**C, kach)
        scores_log = np.append(scores_log, np.mean(kach))
    max_score = max(scores_log)
    for i in range(len(scores_log)):
        if max_score == scores_log[i]:
            C_best = 10**(i-5)

    return max_score, C_best

#Обучение на первичных данных
X_first = scaler.fit_transform(X)
score_first, C_first = logreg(X_first, y)
print(score_first, C_first)

#Обучение на данных без категориальных признаков
X_new = X.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
X_new1 = scaler.fit_transform(X_new)
score_new, C_new = logreg(X_new1, y)
print(score_new, C_new)

#Создание "мешка слов"
N= max(max(X['r1_hero']),max(X['r2_hero']),max(X['r3_hero']),max(X['r4_hero']),max(X['r5_hero']),max(X['d1_hero']),max(X['d2_hero']),max(X['d3_hero']),max(X['d4_hero']),max(X['d5_hero']))
print(N)
X_pick = np.zeros((X.shape[0], N))
for i, match_id in enumerate(X.index):
    for p in range(5):
        X_pick[i, X.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick1 = pd.DataFrame(X_pick)
X_full = pd.concat((X_new, X_pick1), axis=1)


#Обучение с учетом выбранного героя
X_full = scaler.fit_transform(X_full)
score_full, C_full = logreg(X_full, y)
print(score_full, C_full)

#Предсказание на тестовой выборке
best_model = LogisticRegression(penalty='l2', C=C_full, random_state=241)
best_model.fit(X_full, y)

X_new_test = features_test.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
X_pick_test = np.zeros((features_test.shape[0], N))
for i, match_id in enumerate(features_test.index):
    for p in range(5):
        X_pick_test[i, features_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, features_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick_test1 = pd.DataFrame(X_pick_test)
X_full_test = pd.concat((X_new_test, X_pick_test1), axis=1)
X_test = scaler.fit_transform(X_full_test)
final_prob = best_model.predict_proba(X_test)[:, 1]
min_prob = min(final_prob)
for i in range(len(final_prob)):
    if min_prob == final_prob[i]:
        print('min prob =', min_prob)

max_prob = max(final_prob)
for i in range(len(final_prob)):
    if max_prob == final_prob[i]:
        print('max prob =', max_prob)
