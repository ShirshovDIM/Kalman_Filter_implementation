В рамках проекта реализован самописный функционал для фильтра калмана

Среди используемых рабочих классов и функций:
*modules/src/kalman/filters.py*
```python
class BaseKalmanFilter:
    ...

class ExtendedKalmanFilter: 
    ...
```
*modules/src/kalman/utils.py*
```python
def generate_kalman_matrices_from_timeseries_old(...):
    ...

def generate_kalman_matrices_from_timeseries(...):
    ...
```

*modules/src/kalman/plots.py*

Также в рамках работы проведен анализ на основе данных нескольких компаний (Сбербанк, Газпром, Северсталь).
Составлена универсальная модель для приближения значений предсказаний, основываясь на модели внутри дня.

Подробнее в презентации