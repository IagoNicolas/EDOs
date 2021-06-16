#  ___                    _   _
# |_ _|__ _  __ _  ___   | \ | |      Iago Nicolas
#  | |/ _` |/ _` |/ _ \  |  \| |      https://github.com/IagoNicolas
#  | | (_| | (_| | (_) | | |\  |_
# |___\__,_|\__, |\___/  |_| \_(_)    Works with python 3.9.5 64-bit
#           |___/                     @ Thinkpad T480 on 5.12.10-arch1-1, Arch Linux x86_64.

import numpy as np


def feval(funcName, *args):
    return eval(funcName)(*args)


def myFunc(x, y):
    dy = np.zeros((len(y)))
    dy[0] = np.sin(x) + 0.1 * y
    return dy


def exactVal(x, h):
    # Calculates the exact solution, for comparison
    dt = int((x[-1] - x[0]) / h)
    t = [x[0] + i * h for i in range(dt + 1)]
    yexact = []
    for i in range(dt + 1):
        ye = np.exp(+0.1 * t[i] - 0.0990099 * np.sin(t[i]) - 0.990099 * np.cos(t[i]))
        yexact.append(ye)
    return yexact, t


def forwardEuler(func, yinit, x_range, h):
    m = len(yinit)  # Number of ODEs
    n = int((x_range[-1] - x_range[0]) / h)  # Number of sub-intervals

    x = x_range[0]  # Initializes variable x
    y = yinit  # Initializes variable y

    xsol = np.empty(0)  # Creates an empty array for x
    xsol = np.append(xsol, x)  # Fills in the first element of xsol

    ysol = np.empty(0)  # Creates an empty array for y
    ysol = np.append(ysol, y)  # Fills in the initial conditions

    for i in range(n):
        yprime = feval(func, x, y)  # Evaluates dy/dx

        for j in range(m):
            y[j] = y[j] + h * yprime[j]  # Eq. (8.2)

        x += h  # Increase x-step
        xsol = np.append(xsol, x)  # Saves it in the xsol array

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])  # Saves all new y's

    return [xsol, ysol]


def backwardEuler(func, yinit, x_range, h):
    m = len(yinit)
    n = int((x_range[-1] - x_range[0]) / h)

    x = x_range[0]
    y = yinit

    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.empty(0)
    ysol = np.append(ysol, y)

    for i in range(n):
        yprime = feval(func, x + h, y) / (1 + h)

        for j in range(m):
            y[j] = y[j] + h * yprime[j]

        x += h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])  # Saves all new y's

    return [xsol, ysol]


def HeunsMethod(func, yinit, x_range, h):
    m = len(yinit)
    n = int((x_range[-1] - x_range[0]) / h)

    x = x_range[0]
    y = yinit

    # Solution arrays
    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.empty(0)
    ysol = np.append(ysol, y)

    for i in range(n):
        y0prime = feval(func, x, y)

        k1 = y0prime * h

        ypredictor = y + k1

        y1prime = feval(func, x + h, ypredictor)

        for j in range(m):
            y[j] = y[j] + (h / 2) * y0prime[j] + (h / 2) * y1prime[j]

        x = x + h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])  # Saves all new y's

    return [xsol, ysol]


def midpoint(func, yinit, x_range, h):
    m = len(yinit)
    n = int((x_range[-1] - x_range[0]) / h)

    x = x_range[0]
    y = yinit

    # Creates empty arrays for x and y
    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.empty(0)
    ysol = np.append(ysol, y)

    for i in range(n):
        y0prime = feval(func, x, y)

        k1 = y0prime * (h / 2)

        ypredictor = y + k1

        y1prime = feval(func, x + h / 2, ypredictor)

        for j in range(m):
            y[j] = y[j] + h * y1prime[j]

        x = x + h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])

    return [xsol, ysol]


def RK2A(func, yinit, x_range, h):
    m = len(yinit)
    n = int((x_range[-1] - x_range[0]) / h)

    x = x_range[0]
    y = yinit

    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.empty(0)
    ysol = np.append(ysol, y)

    for i in range(n):
        k1 = feval(func, x, y)

        ypredictor = y + k1 * h

        k2 = feval(func, x + h, ypredictor)

        for j in range(m):
            y[j] = y[j] + (h / 2) * (k1[j] + k2[j])

        x = x + h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])

    return [xsol, ysol]


def RK3rdOrder(func, yinit, x_range, h):
    m = len(yinit)
    n = int((x_range[-1] - x_range[0]) / h)

    x = x_range[0]
    y = yinit

    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.empty(0)
    ysol = np.append(ysol, y)

    for i in range(n):
        k1 = feval(func, x, y)

        yp1 = y + k1 * (h / 2)

        k2 = feval(func, x + h / 2, yp1)

        yp2 = y - (k1 * h) + (k2 * 2 * h)

        k3 = feval(func, x + h, yp2)

        for j in range(m):
            y[j] = y[j] + (h / 6) * (k1[j] + 4 * k2[j] + k3[j])

        x = x + h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])

    return [xsol, ysol]


def RK4thOrder(func, yinit, x_range, h):
    m = len(yinit)
    n = int((x_range[-1] - x_range[0]) / h)

    x = x_range[0]
    y = yinit

    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.empty(0)
    ysol = np.append(ysol, y)

    for i in range(n):
        k1 = feval(func, x, y)
        yp2 = y + k1 * (h / 2)
        k2 = feval(func, x + h / 2, yp2)
        yp3 = y + k2 * (h / 2)
        k3 = feval(func, x + h / 2, yp3)
        yp4 = y + k3 * h
        k4 = feval(func, x + h, yp4)

        for j in range(m):
            y[j] = y[j] + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])

        x = x + h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])

    return [xsol, ysol]
