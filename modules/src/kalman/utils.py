import pandas as pd
import numpy as np
from typing import Union, Literal, Optional

## Инициализация + оптимизация параметров
def generate_kalman_matrices_from_timeseries_old(data, state_dimension=1, initial_estimation_error_scale = 1.0):
    """
    Генерация матриц F, H, Q, R, P и вектора x0 из временного ряда данных.

    Аргументы:
        data (pd.DataFrame): Временной ряд данных.

    Возвращает:
        dict: Словарь с матрицами F, H, Q, R, P, x0 и d (размерность измерений).
    """

    T, m = data.shape 
    n = state_dimension  

    # 2. Инициализация F (матрица перехода состояния)
    F = np.zeros((n, n))

    # Оценка временных зависимостей с помощью AR модели для первых m переменных
    for i in range(m):
        if i == 0:
            F[i, i] = 1 

        elif i == 1:
            F[i, i] = 1 
            F[i-1, i] = 1

        elif i > 0:
            F[i-1, i] = 1  # Коэффициенты перехода между состояниями
    if n > m:
        F[m:, m:] = np.eye(n - m)  # Остальные состояния как латентные

    # Матрица наблюдений H (выбираем только наблюдаемые признаки, например, закрытие цен)
    H = np.zeros((m, n))
    for i, col in enumerate(data.columns):
        if 'close' in col.lower():
            H[:, i] = 1

    # Ковариационная матрица шума процесса Q (на основе дисперсий)
    Q = np.diag(data.var())

    # Ковариационная матрица шума измерений R (на основе наблюдаемых данных)
    errors = data - data.mean()
    R = np.array(errors.cov())

    # Ковариационная матрица ошибки оценки P (начальная, единичная или с адаптацией)
    P = initial_estimation_error_scale * np.eye(n)

    # Вектор начального состояния x0 (первая строка данных)
    x0 = data.iloc[0].to_numpy()

    # Размерность измерений
    d = H.shape[0]

    return {
        'F': F,
        'H': H,
        'Q': Q,
        'R': R,
        'P': P,
        'x0': x0,
        'd': d
    }

def generate_kalman_matrices_from_timeseries(time_series_data, state_dimension, process_noise_scale=1.0, measurement_noise_scale=1.0, initial_estimation_error_scale=1.0):
    """
    Генерация матриц F, H, Q, R, P и начального состояния x0 для фильтра Калмана.

    Аргументы:
        time_series_data (ndarray): Временной ряд данных (T временных шагов, m наблюдаемых переменных).
        state_dimension (int): Размерность скрытых состояний (n >= m).
        process_noise_scale (float): Масштаб шума процесса.
        measurement_noise_scale (float): Масштаб шума измерений.
        initial_estimation_error_scale (float): Масштаб начальной ошибки оценки.

    Возвращает:
        tuple: (F, H, Q, R, P, x0).
            F (ndarray): Матрица перехода состояния (n x n).
            H (ndarray): Матрица наблюдений (m x n).
            Q (ndarray): Ковариационная матрица шума процесса (n x n).
            R (ndarray): Ковариационная матрица шума измерений (m x m).
            P (ndarray): Начальная ковариационная матрица ошибки (n x n).
            x0 (ndarray): Начальная оценка состояния (n x 1).
    """
    T, m = time_series_data.shape  # Количество временных шагов и размерность наблюдений
    n = state_dimension            # Размерность состояния

    if n < m:
        raise ValueError("Размерность состояния n должна быть >= размерности наблюдений m.")

    # 1. Анализ временного ряда данных
    Y = time_series_data  # Наблюдаемые переменные

    # 2. Инициализация F (матрица перехода состояния)
    F = np.zeros((n, n))
    # Оценка временных зависимостей с помощью AR модели для первых m переменных
    for i in range(m):
        if i == 0:
            F[i, i] = 0.8  # Пример базового коэффициента для AR(1)
        elif i > 0:
            F[i, i - 1] = 0.5  # Коэффициенты перехода между состояниями
    if n > m:
        F[m:, m:] = np.eye(n - m)  # Остальные состояния как латентные

    # 3. Инициализация H (матрица наблюдений)
    H = np.zeros((m, n))
    H[:m, :m] = np.eye(m)

    # 4. Инициализация Q (ковариация шума процесса)
    Q = process_noise_scale * np.eye(n)

    # 5. Инициализация R (ковариация шума измерений)
    residuals = Y - Y.mean(axis=0)  # Упрощенное представление остаточных ошибок
    R = np.diag(np.var(residuals, axis=0)) + measurement_noise_scale * np.eye(m)

    # 6. Инициализация P (начальная ошибка оценки)
    P = initial_estimation_error_scale * np.eye(n)

    # 7. Инициализация x0 (начальная оценка состояния)
    x0 = np.zeros((n, 1))
    x0[:m, 0] = Y[:5].mean(axis=0)  # Среднее первых 5 наблюдений

    return F, H, Q, R, P, x0

def optimize_kalman_parameters(Z, Q_init, R_init, x_init, P_init, epsilon=1e-6, max_iter=100, method='MLE', alpha=0.01):
    """
    Оптимизация параметров фильтра Калмана.

    Аргументы:
        Z (ndarray): Наблюдения (временной ряд).
        Q_init (ndarray): Начальная оценка ковариации шума процесса.
        R_init (ndarray): Начальная оценка ковариации шума измерений.
        x_init (ndarray): Начальная оценка состояния.
        P_init (ndarray): Начальная ковариация ошибки.
        epsilon (float): Порог сходимости.
        max_iter (int): Максимальное количество итераций.
        method (str): Метод оптимизации ('MLE', 'Gradient', 'EM').
        alpha (float): Скорость обучения (для градиентного спуска).

    Возвращает:
        dict: Оптимизированные параметры Q и R.
    """
    if method == 'MLE':
        # Maximum Likelihood Estimation (MLE) Method
        Q = Q_init
        R = R_init
        x_est = x_init
        P_est = P_init
        log_likelihood_prev = -np.inf
        iteration = 0
        T = len(Z)

        while iteration < max_iter:
            iteration += 1

            # Шаг 1: Выполнение фильтра Калмана с текущими Q и R
            v = []  # Инновации
            S = []  # Ковариации инноваций

            for t in range(T):
                # Предсказание
                x_pred = x_est
                P_pred = P_est + Q

                # Обновление
                K = np.dot(P_pred, np.linalg.inv(P_pred + R))  # Коэффициент Калмана
                x_est = x_pred + np.dot(K, (Z[t] - x_pred))
                P_est = np.dot((np.eye(len(K)) - K), P_pred)

                # Сохранение инноваций и ковариаций
                innovation = Z[t] - x_pred
                innovation_covariance = P_pred + R
                v.append(innovation)
                S.append(innovation_covariance)

            v = np.array(v)
            S = np.array(S)

            # Шаг 2: Вычисление логарифма правдоподобия
            log_likelihood = 0
            d = Z.shape[1] if len(Z.shape) > 1 else 1
            for t in range(T):
                log_likelihood += -0.5 * (np.log(np.linalg.det(S[t])) +
                                          np.dot(v[t].T, np.dot(np.linalg.inv(S[t]), v[t])) +
                                          d * np.log(2 * np.pi))

            # Шаг 3: Проверка сходимости
            if np.abs(log_likelihood - log_likelihood_prev) < epsilon:
                break
            log_likelihood_prev = log_likelihood

            # Шаг 4: Обновление Q и R
            Q = np.sum([np.outer(v[t], v[t]) for t in range(T)], axis=0) / T
            R = np.sum(S, axis=0) / T

        return {
            'Q_opt': Q,
            'R_opt': R
        }

    elif method == 'Gradient':
        # Gradient-Based Optimization
        Q = Q_init
        R = R_init
        x_est = x_init
        P_est = P_init
        log_likelihood_prev = -np.inf
        iteration = 0
        T = len(Z)

        while iteration < max_iter:
            iteration += 1
            v = []
            S = []
            for t in range(T):
                x_pred = x_est
                P_pred = P_est + Q
                K = np.dot(P_pred, np.linalg.inv(P_pred + R))
                x_est = x_pred + np.dot(K, (Z[t] - x_pred))
                P_est = np.dot((np.eye(len(K)) - K), P_pred)
                innovation = Z[t] - x_pred
                innovation_covariance = P_pred + R
                v.append(innovation)
                S.append(innovation_covariance)

            log_likelihood = 0
            d = Z.shape[1] if len(Z.shape) > 1 else 1
            for t in range(T):
                log_likelihood += -0.5 * (np.log(np.linalg.det(S[t])) +
                                          np.dot(v[t].T, np.dot(np.linalg.inv(S[t]), v[t])) +
                                          d * np.log(2 * np.pi))

            if np.abs(log_likelihood - log_likelihood_prev) < epsilon:
                break
            log_likelihood_prev = log_likelihood

            grad_Q = -np.sum([np.linalg.inv(P_pred) @ (np.outer(v[t], v[t]) - Q) @ np.linalg.inv(P_pred) for t in range(T)], axis=0)
            grad_R = -np.sum([np.linalg.inv(S[t]) @ (np.outer(v[t], v[t]) - R) @ np.linalg.inv(S[t]) for t in range(T)], axis=0)

            # Обеспечение совместимости размеров Q и grad_Q, R и grad_R
            if Q.shape != grad_Q.shape:
                grad_Q = grad_Q[:Q.shape[0], :Q.shape[1]]
            if R.shape != grad_R.shape:
                grad_R = grad_R[:R.shape[0], :R.shape[1]]

            Q += alpha * grad_Q
            R += alpha * grad_R

        return {
            'Q_opt': Q,
            'R_opt': R
        }

    elif method == 'EM':
        # Expectation-Maximization Optimization
        Q = Q_init
        R = R_init
        log_likelihood_prev = -np.inf
        iteration = 0
        T = len(Z)

        while iteration < max_iter:
            iteration += 1

            # E-step: Run Kalman filter and smoother
            x_est = x_init
            P_est = P_init
            smoothed_states = []
            smoothed_covariances = []
            for t in range(T):
                x_pred = x_est
                P_pred = P_est + Q
                K = np.dot(P_pred, np.linalg.inv(P_pred + R))
                x_est = x_pred + np.dot(K, (Z[t] - x_pred))
                P_est = np.dot((np.eye(len(K)) - K), P_pred)
                smoothed_states.append(x_est)
                smoothed_covariances.append(P_est)

            smoothed_states = np.array(smoothed_states)
            smoothed_covariances = np.array(smoothed_covariances)

            # M-step: Update Q and R
            Q = np.sum([(smoothed_states[t] - smoothed_states[t - 1]) ** 2 + smoothed_covariances[t] for t in range(1, T)], axis=0) / T
            R = np.sum([(Z[t] - smoothed_states[t]) ** 2 + smoothed_covariances[t] for t in range(T)], axis=0) / T

            # Compute log-likelihood
            log_likelihood = 0
            d = Z.shape[1] if len(Z.shape) > 1 else 1
            for t in range(T):
                log_likelihood += -0.5 * (np.log(np.linalg.det(smoothed_covariances[t])) +
                                          np.dot((Z[t] - smoothed_states[t]).T, np.linalg.inv(smoothed_covariances[t]) @ (Z[t] - smoothed_states[t])) +
                                          d * np.log(2 * np.pi))

            if np.abs(log_likelihood - log_likelihood_prev) < epsilon:
                break
            log_likelihood_prev = log_likelihood

        return {
            'Q_opt': Q,
            'R_opt': R
        }     
