import numpy as np
import sympy as sp

def solve_skateboard():
    # 1. Символьные переменные 
    t = sp.Symbol('t')
    g = 9.81
    
    # Параметры системы (задаются самостоятельно)
    M = 5.0      # Масса скейта
    m1 = 50.0    # Масса туловища
    m2 = 10.0    # Масса рук
    l1 = 0.8     # Расстояние до ЦМ туловища
    l2 = 0.6     # Длина рук
    I1 = 1/12 * m1 * (l1*2)**2  # Момент инерции туловища
    
    # Обобщенные координаты как функции времени
    x = sp.Function('x')(t)
    theta1 = sp.Function('theta1')(t)
    
    # Заданное кинематическое управление для рук (махи с частотой w и амплитудой A)
    A_phi = 0.5   # Амплитуда махов (рад)
    w_phi = 2.0   # Частота махов (рад/с)
    phi = A_phi * sp.sin(w_phi * t)
    
    # Скорости 
    dx = x.diff(t)
    dtheta1 = theta1.diff(t)

    # Центр масс туловища (относительно скейта)
    x_m1 = x + l1 * sp.sin(theta1)
    y_m1 = l1 * sp.cos(theta1)
    
    # Центр масс рук,крепятся к вершине туловища L=2*l1
    L_body = 2 * l1
    x_m2 = x + L_body * sp.sin(theta1) + l2 * sp.sin(theta1 + phi)
    y_m2 = L_body * sp.cos(theta1) + l2 * sp.cos(theta1 + phi)
    
    # Временные производные координат центра масс
    dx_m1 = x_m1.diff(t)
    dy_m1 = y_m1.diff(t)
    dx_m2 = x_m2.diff(t)
    dy_m2 = y_m2.diff(t)
    
    # 2. Энергии системы
    T_skate = 0.5 * M * dx**2
    T_body = 0.5 * m1 * (dx_m1**2 + dy_m1**2) + 0.5 * I1 * dtheta1**2
    T_hands = 0.5 * m2 * (dx_m2**2 + dy_m2**2)  # Считаем материальной точкой 
    
    T = T_skate + T_body + T_hands
    V = m1 * g * y_m1 + m2 * g * y_m2
    
    L = T - V
    
    # 3. Уравнения Лагранжа для x и theta1
    eq_x = sp.diff(sp.diff(L, dx), t) - sp.diff(L, x)
    eq_theta1 = sp.diff(sp.diff(L, dtheta1), t) - sp.diff(L, theta1)
    
    # находим ускорения из системы уравнений
    d2x = x.diff(t, 2)
    d2theta1 = theta1.diff(t, 2)

    sol = sp.solve([eq_x, eq_theta1], (d2x, d2theta1))
    # Нам нужно передавать вектор состояния: [x, dx, theta1, dtheta1] и время t
    state_vars = (x, dx, theta1, dtheta1, t)
    f_d2x = sp.lambdify(state_vars, sol[d2x], 'numpy')
    f_d2theta1 = sp.lambdify(state_vars, sol[d2theta1], 'numpy')
    

    def rhs(t_val, y):
        x_val, dx_val, theta1_val, dtheta1_val = y
        
        ax = f_d2x(x_val, dx_val, theta1_val, dtheta1_val, t_val)
        atheta1 = f_d2theta1(x_val, dx_val, theta1_val, dtheta1_val, t_val)
        
        return np.array([dx_val, ax, dtheta1_val, atheta1], dtype=float)
    
    # 4. Реализация метода Рунге-Кутты 4-го порядка вручную
    t_start, t_end = 0.0, 10.0
    dt = 0.01
    t_steps = np.arange(t_start, t_end, dt)
    
    # Начальное состояние:скейт стоит, туловище чуть отклонено (0.1 рад)
    y = np.array([0.0, 0.0, 0.1, 0.0]) 
    
    history = []
    
    for t_curr in t_steps:
        history.append([t_curr, y[0], y[1], y[2], y[3]])
        
        # Шаги РК4
        k1 = rhs(t_curr, y)
        k2 = rhs(t_curr + dt/2, y + dt*k1/2)
        k3 = rhs(t_curr + dt/2, y + dt*k2/2)
        k4 = rhs(t_curr + dt, y + dt*k3)
        
        y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return np.array(history)

#api решения под FastAPI
if __name__ == "__main__":
    res = solve_skateboard()
    print("Расчет окончен. Шагов:", len(res))
    print("Конечная координата x скейта:", res[-1, 1])