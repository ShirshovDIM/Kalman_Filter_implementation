import numpy as np

class KalmanFilterWithAuxiliaryFactors:
    """
    Класс для реализации фильтра Калмана с дополнительными факторами.
    """
    def __init__(self, F, H, Q, R, X_0, P_0):
        """
        Инициализация фильтра.

        Аргументы:
            F (ndarray): Матрица перехода состояния.
            H (ndarray): Матрица наблюдений.
            Q (ndarray): Ковариационная матрица шума процесса.
            R (ndarray): Ковариационная матрица шума измерений.
            X_0 (ndarray): Начальная оценка состояния.
            P_0 (ndarray): Начальная ковариационная матрица.
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = X_0
        self.P = P_0

    def predict(self):
        """
        Шаг предсказания.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, Y, Z):
        """
        Шаг обновления.

        Аргументы:
            Y (ndarray): Наблюдаемые целевые данные (размерность m_Y).
            Z (ndarray): Наблюдаемые дополнительные данные (размерность m_Z).
        """

        Y_combined = np.hstack((Y, Z))

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (Y_combined - self.H @ self.x)

        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

    def predict_future(self, steps):
        predictions = []
        x_pred = self.x.copy()
        P_pred = self.P.copy()

        for _ in range(steps):
            x_pred = self.F @ x_pred
            P_pred = self.F @ P_pred @ self.F.T + self.Q

            predictions.append(x_pred.copy())

        return np.array(predictions)        

    def get_state(self):
        """
        Возвращает текущее состояние.
        """
        return self.x