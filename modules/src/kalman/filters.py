import numpy as np

class BaseKalmanFilter:
    def __init__(self, F, H, Q, R, P, x0):
        """
        Инициализация фильтра Калмана.

        Аргументы:
            F (ndarray): Матрица перехода состояния (n x n).
            H (ndarray): Матрица наблюдений (m x n).
            Q (ndarray): Ковариационная матрица шума процесса (n x n).
            R (ndarray): Ковариационная матрица шума измерений (m x m).
            P (ndarray): Ковариационная матрица ошибки оценки (n x n).
            x0 (ndarray): Начальная оценка состояния (n x 1).
        """
        self.F = F  # Модель перехода состояния
        self.H = H  # Модель наблюдений
        self.Q = Q  # Ковариация шума процесса
        self.R = R  # Ковариация шума измерений
        self.P = P  # Ковариация ошибки оценки
        self.x = x0  # Начальная оценка состояния
        self.I = np.eye(P.shape[0]) # Начальная матрица ковариаций

    def predict(self):
        """
        Шаг предсказания.
        """
        # Предсказание состояния
        self.x = np.dot(self.F, self.x)

        # Предсказание ковариации ошибки
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """
        Шаг обновления.

        Аргументы:
            z (ndarray): Вектор наблюдений (m x 1).
        """
        # Вычисление коэффициента Калмана
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Ковариация инновации
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Коэффициент Калмана

        # Обновление оценки состояния
        y = z - np.dot(self.H, self.x)  # Остаток измерений
        self.x = self.x + np.dot(K, y)

        # Обновление ковариации ошибки
        self.I = np.eye(self.P.shape[0])  # Единичная матрица
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)

    def get_state(self):
        """
        Получение текущей оценки состояния.

        Возвращает:
            ndarray: Текущая оценка состояния (n x 1).
        """
        return self.x
    
# пока не будет использоваться в примере
class ExtendedKalmanFilter:
    """
    Класс для реализации Расширенного фильтра Калмана (Extended Kalman Filter).
    """
    def __init__(self, F_func, H_func, Q, R, x0, P0):
        """
        Инициализация фильтра.

        Аргументы:
            F_func (callable): Функция перехода состояния f(x, t).
            H_func (callable): Функция наблюдения h(x, t).
            Q (ndarray): Ковариационная матрица шума процесса.
            R (ndarray): Ковариационная матрица шума измерений.
            x0 (ndarray): Начальная оценка состояния.
            P0 (ndarray): Начальная ковариационная матрица ошибки.
        """
        self.F_func = F_func
        self.H_func = H_func
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, t):
        """
        Шаг предсказания.

        Аргументы:
            t (int): Текущий временной шаг.
        """
        try:
            F_jacobian = self.jacobian(self.F_func, self.x, t)  # Якобиан F
            self.x = self.F_func(self.x, t)
            self.P = F_jacobian @ self.P @ F_jacobian.T + self.Q
        except Exception as e:
            raise RuntimeError(f"Ошибка на этапе предсказания: {e}")

    def update(self, z, t):
        """
        Шаг обновления.

        Аргументы:
            z (ndarray): Наблюдения в текущий момент времени.
            t (int): Текущий временной шаг.
        """
        try:
            H_jacobian = self.jacobian(self.H_func, self.x, t)  # Якобиан H
            y = z - self.H_func(self.x, t)  # Инновация
            S = H_jacobian @ self.P @ H_jacobian.T + self.R  # Ковариация инноваций
            K = self.P @ H_jacobian.T @ np.linalg.inv(S)  # Коэффициент Калмана
            self.x = self.x + K @ y
            self.P = (np.eye(len(self.x)) - K @ H_jacobian) @ self.P
        except np.linalg.LinAlgError:
            raise RuntimeError("Сингулярная матрица при вычислении коэффициента Калмана.")
        except Exception as e:
            raise RuntimeError(f"Ошибка на этапе обновления: {e}")

    def jacobian(self, func, x, t, eps=1e-5):
        """
        Численное вычисление Якобиана функции.

        Аргументы:
            func (callable): Функция, для которой вычисляется Якобиан.
            x (ndarray): Точка, в которой вычисляется Якобиан.
            t (int): Текущий временной шаг.
            eps (float): Малое значение для численного дифференцирования.

        Возвращает:
            ndarray: Якобиан функции в точке x.
        """
        n = len(x)
        jac = np.zeros((n, n))
        f_x = func(x, t)
        for i in range(n):
            x_perturbed = x.copy()
            x_perturbed[i] += eps
            f_x_perturbed = func(x_perturbed, t)
            jac[:, i] = (f_x_perturbed - f_x) / eps

        return jac
    
    def predict_future(self, steps, t):
        """
        Предсказание будущих состояний на несколько шагов вперед.
        """
        predictions = []
        x_pred = self.x.copy()
        P_pred = self.P.copy()

        for step in range(steps):
            try:
                F_jacobian = self.jacobian(self.F_func, x_pred, t + step)
                x_pred = self.F_func(x_pred, t + step)
                P_pred = F_jacobian @ P_pred @ F_jacobian.T + self.Q
                predictions.append(x_pred.copy())
            except Exception as e:
                raise RuntimeError(f"Ошибка на этапе предсказания будущего состояния: {e}")

        return np.array(predictions)

    def get_state(self):
        """
        Возвращает текущее состояние.
        """
        return self.x

# todo
class UnscentedKalmanFilter(BaseKalmanFilter):
    def __init__(self, F_func, H_func, Q, R, P, x0):
        """
        Инициализация Неклассического фильтра Калмана.

        Аргументы:
            F_func (function): Нелинейная функция перехода состояния.
            H_func (function): Нелинейная модель наблюдений.
            Q (ndarray): Ковариация шума процесса.
            R (ndarray): Ковариация шума измерений.
            P (ndarray): Ковариация ошибки оценки.
            x0 (ndarray): Начальная оценка состояния.
        """
        super().__init__(None, None, Q, R, P, x0)
        self.F_func = F_func
        self.H_func = H_func

    def predict(self):
        """
        Шаг предсказания Неклассического фильтра Калмана.
        """
        sigma_points = self.generate_sigma_points()
        sigma_points_pred = [self.F_func(sp) for sp in sigma_points]
        self.x = np.mean(sigma_points_pred, axis=0)
        self.P = np.cov(np.array(sigma_points_pred).T) + self.Q

    def update(self, z):
        """
        Шаг обновления Неклассического фильтра Калмана.

        Аргументы:
            z (ndarray): Вектор наблюдений.
        """
        sigma_points = self.generate_sigma_points()
        sigma_points_obs = [self.H_func(sp) for sp in sigma_points]
        z_pred = np.mean(sigma_points_obs, axis=0)
        P_zz = np.cov(np.array(sigma_points_obs).T) + self.R
        P_xz = np.cov(np.array([sp - self.x for sp in sigma_points]).T,
                      np.array([sp_obs - z_pred for sp_obs in sigma_points_obs]).T)
        K = np.dot(P_xz, np.linalg.inv(P_zz))
        self.x = self.x + np.dot(K, (z - z_pred))
        self.P = self.P - np.dot(K, P_zz).dot(K.T)

    def generate_sigma_points(self, alpha=1e-3, beta=2, kappa=0):
        """
        Генерация сигма-точек для Неклассического фильтра Калмана.

        Аргументы:
            alpha (float): Параметр разброса сигма-точек.
            beta (float): Параметр распределения.
            kappa (float): Вторичный параметр масштабирования.

        Возвращает:
            list
        """
        n = self.x.shape[0]
        lambda_ = alpha ** 2 * (n + kappa) - n
        sigma_points = [self.x]
        sqrt_P = np.linalg.cholesky((n + lambda_) * self.P)
        for i in range(n):
            sigma_points.append(self.x + sqrt_P[:, i])
            sigma_points.append(self.x - sqrt_P[:, i])
        return sigma_points

# Если нужно будет строить итеративный прогноз
class IterativeKalmanFilter(BaseKalmanFilter):
    def __init__(self, F, H, Q, R, P, x0, max_iterations=10, tolerance=1e-5):
        """
        Итеративный фильтра Калмана.

        Аргументы:
            F (ndarray): Матрица перехода состояния.
            H (ndarray): Матрица наблюдений.
            Q (ndarray): Ковариация шума процесса.
            R (ndarray): Ковариация шума измерений.
            P (ndarray): Ковариация ошибки оценки.
            x0 (ndarray): Начальная оценка состояния.
            max_iterations (int): Максимальное количество итераций для сходимости.
            tolerance (float): Допустимая ошибка для сходимости.
        """
        super().__init__(F, H, Q, R, P, x0)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.I = np.eye(self.P.shape[0])

    def update(self, z):
        """
        итеративный шаг обновления фильтра Калмана.

        Аргументы:
            z (ndarray): Вектор наблюдений.
        """
        for _ in range(self.max_iterations):
            # Прогноз наблюдений
            z_pred = np.dot(self.H, self.x)

            # Вычисление остатка
            y = z - z_pred

            # Вычисление ковариации инновации
            S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

            # Вычисление коэффициента Калмана
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

            # Обновление оценки состояния
            x_new = self.x + np.dot(K, y)

            # Проверка на сходимость
            if np.linalg.norm(x_new - self.x) < self.tolerance:
                self.x = x_new
                break

            self.x = x_new

        # Обновление ковариации ошибки
        self.I = np.eye(self.P.shape[0])
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)
