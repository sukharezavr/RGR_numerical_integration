import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre

# ПРОСТЕЙШИЕ ФУНКЦИИ

def rectangles(f, n, a, b):
    h = (b - a) / n
    result = 0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        result += f(x_mid)
    return h * result


def trapezoid(f, n, a, b):
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    return h * ((y[0] + y[-1]) / 2 + sum(y[1:-1]))


def simpson(f, n, a, b):
    if n % 2 != 0:
        n += 1  # чтобы точно было четным
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    return h / 3 * (y[0] + y[-1] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]))


def three_eighths(f, n, a, b):
    if n % 3 != 0:
        n = ((n + 2) // 3) * 3  # чтобы точно было кратным 3
    h = (b - a) / n
    result = 0
    for i in range(0, n, 3):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        x2 = a + (i + 2) * h
        x3 = a + (i + 3) * h
        result += (3 * h / 8) * (f(x0) + 3 * f(x1) + 3 * f(x2) + f(x3))
    return result

# КВАДРАТУРНЫЕ ФОРМУЛЫ ГАУССОВСКОГО ТИПА

def gauss_quadrature_auto(f, n_nodes, a, b):
    """Гаусс для любого n_nodes (автоматические узлы и веса)"""
    nodes, weights = legendre.leggauss(n_nodes)
    mid = (a + b) / 2
    half_len = (b - a) / 2
    result = 0
    for ti, Ai in zip(nodes, weights):
        x = mid + half_len * ti
        result += Ai * f(x)
    return half_len * result


def chebyshev_quadrature(f, n_nodes, a, b):
    """Чебышев, n_nodes = 2,3,4,5 (все веса = 2/n)"""
    tables = {
        2: [-0.577350, 0.577350],
        3: [-0.707107, 0.0, 0.707107],
        4: [-0.794654, -0.187592, 0.187592, 0.794654],
        5: [-0.832497, -0.374541, 0.0, 0.374541, 0.832497],
    }
    if n_nodes not in tables:
        raise ValueError(f"Чебышев: n_nodes={n_nodes} не поддерживается")
    t = tables[n_nodes]
    A = [2.0 / n_nodes] * n_nodes
    mid = (a + b) / 2
    half_len = (b - a) / 2
    result = 0
    for ti, Ai in zip(t, A):
        x = mid + half_len * ti
        result += Ai * f(x)
    return half_len * result


def radau_quadrature(f, n_nodes, a, b):
    """Радо, n_nodes = 2,3,4,5 (фиксирован t = -1)"""
    # ВНИМАНИЕ: n_nodes здесь — общее число узлов (включая фиксированный)
    tables = {
        2: ([-1.0, 0.333333], [0.5, 1.5]),
        3: ([-1.0, -0.289898, 0.689898], [0.222222, 1.024972, 0.752806]),
        4: ([-1.0, -0.575319, 0.181066, 0.822824], [0.125, 0.657689, 0.776387, 0.440924]),
        5: ([-1.0, -0.802929, -0.390928, 0.124050, 0.603973, 0.920380],
            [0.055556, 0.319640, 0.485387, 0.520927, 0.416901, 0.201588]),
    }
    if n_nodes not in tables:
        raise ValueError(f"Радо: n_nodes={n_nodes} не поддерживается")
    t, A = tables[n_nodes]
    mid = (a + b) / 2
    half_len = (b - a) / 2
    result = 0
    for ti, Ai in zip(t, A):
        x = mid + half_len * ti
        result += Ai * f(x)
    return half_len * result


def lobatto_quadrature(f, n_nodes, a, b):
    """Лобатто, n_nodes = 2,3,4,5 (фиксированы t = -1 и t = 1)"""
    tables = {
        2: ([-1.0, 1.0], [1.0, 1.0]),
        3: ([-1.0, 0.0, 1.0], [1/3, 4/3, 1/3]),
        4: ([-1.0, -0.447214, 0.447214, 1.0], [1/6, 5/6, 5/6, 1/6]),
        5: ([-1.0, -0.654654, 0.0, 0.654654, 1.0],
            [0.1, 0.544444, 0.711111, 0.544444, 0.1]),
    }
    if n_nodes not in tables:
        raise ValueError(f"Лобатто: n_nodes={n_nodes} не поддерживается")
    t, A = tables[n_nodes]
    mid = (a + b) / 2
    half_len = (b - a) / 2
    result = 0
    for ti, Ai in zip(t, A):
        x = mid + half_len * ti
        result += Ai * f(x)
    return half_len * result

#ОШИБКА РУНГЕ

def runge_error(I_n, I_2n, p):
    return abs(I_n - I_2n) / (2 ** p - 1)

# ПОСЛЕДОВАТЕЛЬНО РАЗБИВАЕМ НА ВСЕ БОЛЬШЕЕ ЧИСЛО КУСКОВ

def adaptive_integrate(method, f, a, b, eps, p, method_name):
    n = 4
    max_iter = 20

    print(f"\nМетод: {method_name}")
    print(f"{'n':>5} {'h':>10} {'I':>12} {'error':>12}")

    for iteration in range(max_iter):
        h = (b - a) / n
        I_n = method(f, n, a, b)
        I_2n = method(f, 2 * n, a, b)

        error = runge_error(I_n, I_2n, p)

        print(f"{n:5d} {h:10.6f} {I_2n:12.8f} {error:12.2e}")

        if error < eps:
            print(f"Достигнута точность {eps} за {iteration + 1} итераций")
            return I_2n, n

        n *= 2

    print("Достигнуто максимальное число итераций")
    return I_2n, n


#ФУНКЦИИ КОТОРЫЕ ИНТЕГРИРУЕМ
def f_power(x):
    return x**2

def f_poisson(x):
    return math.exp(-x * x)

def f_abs(x):
    return abs(x)

def f_sqrt_abs(x):
    return math.sqrt(abs(x))

#НАЧАЛЬНЫЕ ДАННЫЕ ДЛЯ РАЗНЫХ ФУНКЦИЙ

test_tasks = {
    'x²': {
        'f': f_power,
        'a': 0, 'b': 1,
        'exact': 1/3,
        'note': 'Многочлен 2 степени'
    },
    'Интеграл Пуассона': {
        'f': f_poisson,
        'a': -2, 'b': 2,
        'exact': 1.77245,
        'note': 'Гладкая, неберущаяся'
    },
    'Корень': {
    'f': f_sqrt_abs,
    'a': -1, 'b': 1,
    'exact': 4/3,
    'note': 'Разрыв производной (корень)'
    }
}

#АНАЛИЗ

def analysis():
    print("\nВЛИЯНИЕ ТОЧНОСТИ НА ЧИСЛО РАЗБИЕНИЙ")
    print("-" * 50)

    f_test = f_poisson
    a, b = -2, 2
    eps_list = [1e-3, 1e-6, 1e-9]
    for task_name, task in test_tasks.items():
        print(f"\nФункция: {task_name} — {task['note']}")
        print(f"Отрезок [{task['a']}, {task['b']}]")
        print("-" * 60)
        print(f"{'Метод':<20} {'ε=1e-3':>12} {'ε=1e-6':>12} {'ε=1e-9':>12}")
        print("-" * 60)

        for method, p, name in methods:
            row = f"{name:<20}"
            for eps in eps_list:
                try:
                    _, n = adaptive_integrate(method, f_test, a, b, eps, p, name)
                    row += f"{n:12d}"
                except:
                    row += f"{'---':>12}"
            print(row)

    print("\n\nВЛИЯНИЕ ГЛАДКОСТИ ФУНКЦИИ НА ТОЧНОСТЬ")
    h_fixed = 0.1
    print(f"При фиксированном шаге h = {h_fixed}")

    print("\nФактическая погрешность для разных функций")
    print("=" * 70)
    print(f"{'Метод':<20} {'x²':>15} {'Пуассон':>15} {'√|x|':>15}")
    print("-" * 100)

    for method, p, name in methods:
        row = f"{name:<20}"
        for task_name, task in test_tasks.items():
            try:
                # Вычисляем n по h
                L = task['b'] - task['a']
                n = int(round(L / h_fixed))
                if n < 4:
                    n = 4

                # Корректировка n под метод
                if method == simpson and n % 2 != 0:
                    n += 1
                elif method == three_eighths and n % 3 != 0:
                    n = ((n + 2) // 3) * 3

                I = method(task['f'], n, task['a'], task['b'])
                error = abs(task['exact'] - I)
                row += f"{error:15.2e}"
            except:
                row += f"{'---':>15}"
        print(row)

    print("\n\nСРАВНЕНИЕ ТЕОРЕТИЧЕСКОЙ И ФАКТИЧЕСКОЙ ПОГРЕШНОСТИ")
    print("-" * 50)

    f_test = f_poisson
    a, b = -2, 2
    exact = 1.77245
    n_list = [4, 8, 16, 32, 64]

    print("\nПогрешность при разных n (интеграл Пуассона)")
    print("=" * 80)
    print(f"{'Метод':<20} {'n':>5} {'h':>10} {'Факт. ошибка':>15} {'Теор. оценка':>15}")
    print("-" * 80)

    for method, p, name in methods:
        errors = []
        h_values = []

        for n in n_list:
            try:
                if method == simpson and n % 2 != 0:
                    n_used = n + 1
                elif method == three_eighths and n % 3 != 0:
                    n_used = ((n + 2) // 3) * 3
                else:
                    n_used = n

                h = (b - a) / n_used
                I = method(f_test, n_used, a, b)
                actual_error = abs(exact - I)

                if p == 2:
                    theo_error = (b - a) / 12 * 2 * h ** 2
                else:
                    theo_error = (b - a) / 180 * 12 * h ** 4

                errors.append(actual_error)
                h_values.append(h)

                print(f"{name:<20} {n_used:5d} {h:10.6f} {actual_error:15.2e} {theo_error:15.2e}")

            except:
                print(f"{name:<20} {n:5d} {'---':>10} {'---':>15} {'---':>15}")


def analysis_new():
    print("\n" + "=" * 70)
    print("ЧИСЛЕННЫЙ АНАЛИЗ ДЛЯ МЕТОДОВ ГАУССОВСКОГО ТИПА")
    print("=" * 70)

    new_methods = [
        (gauss_quadrature_auto, "Гаусс (авто)"),
        (chebyshev_quadrature, "Чебышев"),
        (radau_quadrature, "Радо"),
        (lobatto_quadrature, "Лобатто"),
    ]

    k_values = [2, 3, 4, 5, 10, 15, 20]  # для Гаусса (авто)
    k_table = [2, 3, 4, 5]  # для табличных методов

    for task_name, task in test_tasks.items():
        print(f"\n{'=' * 70}")
        print(f"Функция: {task_name} — {task['note']}")
        print(f"Отрезок [{task['a']}, {task['b']}]")
        print(f"Точное значение: {task['exact']:.8f}")
        print("=" * 70)

        # Шапка таблицы
        print(f"{'Метод':<20} {'k':>5} {'Приближение':>15} {'Ошибка':>15}")
        print("-" * 60)

        for method, name in new_methods:
            # Определяем, какие k использовать для этого метода
            if name == "Гаусс (авто)":
                ks = k_values
            else:
                ks = k_table

            for k in ks:
                try:
                    I = method(task['f'], k, task['a'], task['b'])
                    error = abs(task['exact'] - I)
                    print(f"{name:<20} {k:5d} {I:15.8f} {error:15.2e}")
                except Exception as e:
                    print(f"{name:<20} {k:5d} {'---':>15} {'ошибка':>15}")
            print("-" * 60)

def plot_error():
    n_values = [4, 8, 16, 32, 64, 128, 256, 512]

    for task_name, task in test_tasks.items():
        plt.figure(figsize=(12, 8))

        for method, p, method_name in methods:
            errors = []
            valid_n = []

            for n in n_values:
                try:
                    # Корректировка n
                    if method == simpson and n % 2 != 0:
                        n_used = n + 1
                    elif method == three_eighths and n % 3 != 0:
                        n_used = ((n + 2) // 3) * 3
                    else:
                        n_used = n

                    I = method(task['f'], n_used, task['a'], task['b'])
                    error = abs(task['exact'] - I)

                    if error > 1e-16:
                        errors.append(error)
                        valid_n.append(n_used)

                except Exception as e:
                    print(f"Ошибка для {method_name} при n={n}: {e}")
                    continue

            if errors:
                plt.semilogy(valid_n, errors, 'o-', label=method_name,
                             linewidth=2, markersize=4)

        plt.xlabel('число разбиений n')
        plt.ylabel('фактическая погрешность (лог. шкала)')
        plt.title(task_name)
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(f'error_plot_{task_name}.png', dpi=150, bbox_inches='tight')
        plt.show()


def plot_error_simple():
    # Диапазоны
    n_newton = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    n_gauss = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    n_table = [2, 3, 4, 5]

    methods_newton = [
        (rectangles, "Прямоугольники"),
        (trapezoid, "Трапеции"),
        (simpson, "Симпсон"),
        (three_eighths, "Три восьмых"),
    ]

    table_methods = [
        (chebyshev_quadrature, "Чебышев"),
        (radau_quadrature, "Радо"),
        (lobatto_quadrature, "Лобатто"),
    ]

    for task_name, task in test_tasks.items():
        if task_name == 'x²':
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Левый график: Ньютон–Котес
        ax1 = axes[0]
        for method, name in methods_newton:
            errors = []
            xs = []
            for n in n_newton:
                try:
                    if method == simpson and n % 2 != 0:
                        continue
                    if method == three_eighths and n % 3 != 0:
                        continue
                    I = method(task['f'], n, task['a'], task['b'])
                    error = abs(task['exact'] - I)
                    if error > 1e-16:
                        errors.append(error)
                        xs.append(n)
                except:
                    continue
            if errors:
                ax1.semilogy(xs, errors, 'o-', label=name, linewidth=1.5, markersize=4)
        ax1.set_xlabel('число отрезков n')
        ax1.set_ylabel('реальная ошибка (лог. шкала)')
        ax1.set_title(f'{task_name} — методы Ньютона–Котеса')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Правый график: Гаусс, Чебышев, Радо, Лобатто
        ax2 = axes[1]

        # Гаусс (авто)
        errors = []
        xs = []
        for k in n_gauss:
            try:
                I = gauss_quadrature_auto(task['f'], k, task['a'], task['b'])
                error = abs(task['exact'] - I)
                if error > 1e-16:
                    errors.append(error)
                    xs.append(k)
            except:
                continue
        if errors:
            ax2.semilogy(xs, errors, 's-', label="Гаусс", linewidth=2, markersize=5)

        # Чебышев, Радо, Лобатто
        markers = ['d-', '^-', 'v-']
        for (method, name), marker in zip(table_methods, markers):
            errors = []
            xs = []
            for k in n_table:
                try:
                    I = method(task['f'], k, task['a'], task['b'])
                    error = abs(task['exact'] - I)
                    if error > 1e-16:
                        errors.append(error)
                        xs.append(k)
                except:
                    continue
            if errors:
                ax2.semilogy(xs, errors, marker, label=name, linewidth=2, markersize=6)

        ax2.set_xlabel('число узлов k')
        ax2.set_ylabel('реальная ошибка (лог. шкала)')
        ax2.set_title(f'{task_name} — методы Гауссовского типа')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'error_plot_split_{task_name}.png', dpi=150, bbox_inches='tight')
        plt.show()

methods = [
    (rectangles, 2, "Прямоугольники"),
    (trapezoid, 2, "Трапеции"),
    (simpson, 4, "Симпсон"),
    (three_eighths, 4, "Три восьмых")
]

analysis()
plot_error()
plot_error_simple()
analysis_new()
