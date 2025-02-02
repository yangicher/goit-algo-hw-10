import pulp as plp
import numpy as np
import scipy.integrate as integrate
import monte_carlo

model = plp.LpProblem("", plp.LpMaximize)
L = plp.LpVariable("Lemonade", 0, None, cat="Integer")
J = plp.LpVariable("Juice", 0, None, cat="Integer")

model += L + J, "Profit"

model += 2 * L + 1 * J <= 100, "water"
model += L <= 100 / 2, "lemonade_w"
model += L <= 50, "sugar"
model += L <= 30, "lime_juice"
model += J <= 100, "juice_w"
model += J <= 40 / 2, "fruit_puree"

model.solve()
print(f"Status: {plp.LpStatus[model.status]}")
print(f"Lemonade: {plp.value(L)}")
print(f"Juice: {plp.value(J)}")
print(f"Max total = {plp.value(model.objective)}")

def monte_carlo_integrate(func, a, b, y_min, y_max, num_points):
    x = np.random.uniform(a, b, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    under_curve = np.sum(y < func(x))
    area = (b - a) * (y_max - y_min) * (under_curve / num_points)
    return area

def write_conclusion(result, mc_result, err):
    with open("README.md", "w", encoding="utf-8") as file:
        file.write("# Conclusions \n\n")
        file.write(f"Результат функції quad {result} і її похибка {err}\n")
        file.write(f"Результат алгоритму Монте-Карло {mc_result}\n")
        file.write(f"Різниця між цими результатами = {abs(result - mc_result)}\n\n")

        file.write("🔹 Метод Монте-Карло дає досить точний результат з малою похибкою (~0.1%). \n")
        file.write("🔹 Функція quad є більш точною, оскільки використовує чисельні методи для точного інтегрування.\n")
        file.write("🔹 Чим більше випадкових точок, тим точніший метод Монте-Карло.\n")

if __name__ == "__main__":
    a = -2
    b = 2
    y_min = 0
    y_max = 4
    x = np.linspace(-2, 2, 100)
    y = monte_carlo.f(x)
    result, err = integrate.quad(monte_carlo.f, a, b)
    mc_result = monte_carlo_integrate(monte_carlo.f, a, b, y_min, y_max, 1_000_000)

    write_conclusion(result, mc_result, err)