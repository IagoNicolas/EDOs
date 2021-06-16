from assets import questao_1, questao_2, questao_3, questao_4, questao_5
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from multiprocessing import Process

######################################
############# Questão 1 ##############
######################################


def q1_Euler():
    h = 0.2
    x = np.array([0.0, 20.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_1.forwardEuler("myFunc", yinit, x, h)

    yexact, t = questao_1.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 1 - Euler #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 1 - Euler Direto")
    plt.legend(["Euler Direto", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q1_RK():
    h = 0.2
    x = np.array([0.0, 20.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_1.RK4thOrder("myFunc", yinit, x, h)

    yexact, t = questao_1.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 1 - Runge-Kutta #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 1 - Runge-Kutta")
    plt.legend(["Runge-Kutta de ordem 4", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q1_lsoda():
    def sciFunc(y, x):
        dy = np.zeros(len(y))
        dy = -0.3 * y
        return dy

    yinit = 1  # Valor inicial

    h = 0.2
    x = np.array([0.0, 20.0])
    yinit = np.array([1.0])

    t = np.linspace(x[0], x[1], int(x[1] / 0.2) + 1)
    y = odeint(sciFunc, yinit, t)

    yexact, t = questao_1.exactVal(x, h)

    plt.plot(t, y, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 1 - LSODA")
    plt.legend(["LSODA", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


######################################
############# Questão 2 ##############
######################################


def q2_Euler():
    h = 0.2
    x = np.array([0.0, 100.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_2.forwardEuler("myFunc", yinit, x, h)

    yexact, t = questao_2.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 2 - Euler #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 2 - Euler Direto")
    plt.legend(["Euler Direto", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q2_RK():
    h = 0.2
    x = np.array([0.0, 100.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_2.RK4thOrder("myFunc", yinit, x, h)

    yexact, t = questao_2.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 2 - Runge-Kutta #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 2 - Runge-Kutta")
    plt.legend(["Runge-Kutta de ordem 4", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q2_lsoda():
    def sciFunc(y, x):
        dy = np.zeros(len(y))
        dy = +0.3 * y
        return dy

    yinit = 1  # Valor inicial

    h = 0.2
    x = np.array([0.0, 100.0])
    yinit = np.array([1.0])

    t = np.linspace(x[0], x[1], int(x[1] / 0.2) + 1)
    y = odeint(sciFunc, yinit, t)

    yexact, t = questao_2.exactVal(x, h)

    plt.plot(t, y, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 2 - LSODA")
    plt.legend(["LSODA", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


######################################
############# Questão 3 ##############
######################################


def q3_Euler():
    h = 0.2
    x = np.array([0.0, 100.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_3.forwardEuler("myFunc", yinit, x, h)

    yexact, t = questao_3.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 3 - Euler #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 3 - Euler Direto")
    plt.legend(["Euler Direto", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q3_RK():
    h = 0.2
    x = np.array([0.0, 100.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_3.RK4thOrder("myFunc", yinit, x, h)

    yexact, t = questao_3.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 3 - Runge-Kutta #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 3 - Runge-Kutta")
    plt.legend(["Runge-Kutta de ordem 4", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q3_lsoda():
    def sciFunc(y, x):
        dy = np.zeros(len(y))
        dy[0] = np.sin(x) - 0.1 * y
        return dy

    yinit = 1  # Valor inicial

    h = 0.2
    x = np.array([0.0, 100.0])
    yinit = np.array([1.0])

    t = np.linspace(x[0], x[1], int(x[1] / 0.2) + 1)
    y = odeint(sciFunc, yinit, t)

    yexact, t = questao_3.exactVal(x, h)

    plt.plot(t, y, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 3 - LSODA")
    plt.legend(["LSODA", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


######################################
############# Questão 4 ##############
######################################


def q4_Euler():
    h = 0.2
    x = np.array([0.0, 10.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_4.forwardEuler("myFunc", yinit, x, h)

    yexact, t = questao_4.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 4 - Euler #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 4 - Euler Direto")
    plt.legend(["Euler Direto", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q4_RK():
    h = 0.2
    x = np.array([0.0, 10.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_4.RK4thOrder("myFunc", yinit, x, h)

    yexact, t = questao_4.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 4 - Runge-Kutta #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 4 - Runge-Kutta")
    plt.legend(["Runge-Kutta de ordem 4", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q4_lsoda():
    def sciFunc(y, x):
        dy = np.zeros(len(y))
        dy[0] = np.sin(x) + 0.1 * y
        return dy

    yinit = 1  # Valor inicial

    h = 0.2
    x = np.array([0.0, 10.0])
    yinit = np.array([1.0])

    t = np.linspace(x[0], x[1], int(x[1] / 0.2) + 1)
    y = odeint(sciFunc, yinit, t)

    yexact, t = questao_4.exactVal(x, h)

    plt.plot(t, y, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 4 - LSODA")
    plt.legend(["LSODA", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


######################################
############# Questão 5 ##############
######################################


def q5_Euler():
    h = 0.2
    x = np.array([0.0, 20.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_5.forwardEuler("myFunc", yinit, x, h)

    yexact, t = questao_5.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 5 - Euler #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 5 - Euler Direto")
    plt.legend(["Euler Direto", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q5_RK():
    h = 0.2
    x = np.array([0.0, 20.0])
    yinit = np.array([1.0])

    [ts, ys] = questao_5.RK4thOrder("myFunc", yinit, x, h)

    yexact, t = questao_5.exactVal(x, h)

    diff = ys - yexact
    difference_array = np.subtract(yexact, ys)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("##### Equação 5 - Runge-Kutta #####")
    print("Maximum difference =", np.max(abs(diff)))
    print("Mean squared error", mse)

    plt.plot(ts, ys, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 5 - Runge-Kutta")
    plt.legend(["Runge-Kutta de ordem 4", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


def q5_lsoda():
    def sciFunc(y, x):
        dy = np.zeros(len(y))
        dy = 3 * y * (1 - np.power((x / 10), 2))
        return dy

    yinit = 1  # Valor inicial

    h = 0.2
    x = np.array([0.0, 20.0])
    yinit = np.array([1.0])

    t = np.linspace(x[0], x[1], int(x[1] / 0.2) + 1)
    y = odeint(sciFunc, yinit, t)

    yexact, t = questao_5.exactVal(x, h)

    plt.plot(t, y, "r")
    plt.plot(t, yexact, "b")
    plt.xlim(x[0], x[1])
    plt.title("Eq 5 - LSODA")
    plt.legend(["LSODA", "Solução exata"], loc=2)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # As questoes estao sendo resolvidas em paralelo, a saída
    # é mostrada na ordem em que os calculos são finalizados.
    q1_1 = Process(target=q1_Euler, args=()).start()
    q1_2 = Process(target=q1_RK, args=()).start()
    q1_3 = Process(target=q1_lsoda, args=()).start()
    q2_1 = Process(target=q2_Euler, args=()).start()
    q2_2 = Process(target=q2_RK, args=()).start()
    q2_3 = Process(target=q2_lsoda, args=()).start()
    q3_1 = Process(target=q3_Euler, args=()).start()
    q3_2 = Process(target=q3_RK, args=()).start()
    q3_3 = Process(target=q3_lsoda, args=()).start()
    q4_1 = Process(target=q4_Euler, args=()).start()
    q4_2 = Process(target=q4_RK, args=()).start()
    q4_3 = Process(target=q4_lsoda, args=()).start()
    q5_1 = Process(target=q5_Euler, args=()).start()
    q5_2 = Process(target=q5_RK, args=()).start()
    q5_3 = Process(target=q5_lsoda, args=()).start()
    pass
