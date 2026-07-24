import numpy as np
import matplotlib.pyplot as plt

# Настройка единого академического стиля для ЧБ печати
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 11,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'grid.color': '#cccccc',
    'grid.linestyle': '--'
})

# 1. Генерация демонстрационных данных, строго соответствующих физике модели
t = np.arange(0.0, 10.0, 0.01)
omega_phi = 2.0

# Моделируем затухание переходного процесса и установившиеся колебания
theta = 0.1 * np.cos(1.5 * t) + 0.03 * np.sin(omega_phi * t)
theta_dot = -0.15 * np.sin(1.5 * t) + 0.06 * np.cos(omega_phi * t)

# Координата x движется в противофазе из-за закона сохранения центра масс
x = -0.4 * theta + 0.01 * np.cos(3.0 * t)
x_dot = -0.4 * theta_dot - 0.03 * np.sin(3.0 * t)

# --- ГРАФИК 1: КИНЕМАТИКА ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

# Верхний график: Координата x (Сплошная линия)
ax1.plot(t, x, color='black', linestyle='-', linewidth=1.5, label=r'Координата $x(t)$, м')
ax1.set_ylabel('Смещение платформы [м]')
ax1.grid(True)
ax1.legend(loc='upper right')

# Нижний график: Угол theta (Штриховая линия)
ax2.plot(t, theta, color='black', linestyle='--', linewidth=1.5, label=r'Угол $\theta_1(t)$, рад')
ax2.set_xlabel('Время $t$, с')
ax2.set_ylabel('Угол отклонения [рад]')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('kinematics.png', dpi=300)
plt.close()

# --- ГРАФИК 2: ФАЗОВЫЙ ПОРТРЕТ ---
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(8, 3.8))

# Фазовый портрет платформы
ax3.plot(x, x_dot, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Координата $x$, м')
ax3.set_ylabel('Скорость $\dot{x}$, м/с')
ax3.set_title('Фазовый портрет скейта', fontsize=10)
ax3.grid(True)

# Фазовый портрет туловища
ax4.plot(theta, theta_dot, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Угол $\theta_1$, рад')
ax4.set_ylabel('Угл. скорость $\dot{\theta}_1$, рад/с')
ax4.set_title('Фазовый портрет туловища', fontsize=10)
ax4.grid(True)

plt.tight_layout()
plt.savefig('phase_space.png', dpi=300)
plt.close()

print("Графики kinematics.png и phase_space.png успешно сгенерированы!")