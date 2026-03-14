import math
import matplotlib.pyplot as plt

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
    '√|x|': {
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

    plt.figure(figsize=(10, 6))

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

        if errors:
            plt.loglog(h_values, errors, 'o-', label=name, linewidth=2)

    plt.xlabel('шаг h')
    plt.ylabel('погрешность')
    plt.title('Зависимость погрешности от шага (интеграл Пуассона)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('error_vs_h.png', dpi=150, bbox_inches='tight')
    plt.show()


methods = [
    (rectangles, 2, "Прямоугольники"),
    (trapezoid, 2, "Трапеции"),
    (simpson, 4, "Симпсон"),
    (three_eighths, 4, "Три восьмых")
]

analysis()