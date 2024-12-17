import numpy as np

class KalmanFilterWithAuxiliaryFactors:
    """
    Реализация расширенного Калмановского фильтра для оценки латентных факторов.
    """
    def __init__(self, A, H, F_0, P_0, Q, R):
        """
        Инициализация фильтра.

        Аргументы:
            A (ndarray): Матрица перехода состояния (k x k).
            H (ndarray): Матрица наблюдений (m x k).
            F_0 (ndarray): Начальная оценка латентных факторов (k x 1).
            P_0 (ndarray): Начальная ковариационная матрица (k x k).
            Q (ndarray): Ковариационная матрица шума процесса (k x k).
            R (ndarray): Ковариационная матрица шума измерений (m x m).
        """
        self.A = A
        self.H = H
        self.x = F_0.flatten()  # Текущее состояние (латентные факторы)
        self.P = P_0  # Текущая ковариационная матрица
        self.Q = Q
        self.R = R

    def predict(self):
        """
        Шаг предсказания.

        Возвращает:
            tuple: Предсказанное состояние (x_pred) и ковариационная матрица (P_pred).
        """
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P

    def update(self, y):
        """
        Шаг обновления.

        Аргументы:
            y (ndarray): Наблюдаемое значение в текущий момент времени (размерность m).

        Возвращает:
            tuple: Обновленное состояние (x) и ковариационная матрица (P).
        """
        # Вычисление коэффициента Калмана
        S = self.H @ self.P @ self.H.T + self.R  # Ковариация инноваций
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Коэффициент Калмана

        # Обновление состояния
        innovation = y - self.H @ self.x  # Инновация
        self.x = self.x + K @ innovation

        # Обновление ковариационной матрицы
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        return self.x, self.P