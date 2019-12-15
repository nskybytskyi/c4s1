#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

λ: float = 220.0
c: float = 890.0
ρ: float = 2_700.0
α: float = λ / (c * ρ)
γ: float = 300.0
end_time: int = 3_600
radius: float = 0.5
environ_temperature: float = -20.0
initial_temperature: float = +20.0

N: int = 50
M: int = end_time * 4
h: float = radius / N
τ: float = end_time / M
σ: float = 0.5 - h ** 2 / (12 * τ)

print(f"h = {h:.7f} (h^4 = {h ** 4:.7f})")
print(f"τ = {τ:.7f} (τ^2 = {τ ** 2:.7f})")
print(f"σ = {σ:.7f}")

β_2: float = γ / (c * ρ)
μ_2: float = γ * environ_temperature / (c * ρ)
q: float = γ / (c * ρ)

A_ = np.zeros((N + 1, N + 1))
b_ = np.zeros(N + 1)


def p_i(i: int):
    return λ / (c * ρ) * (h * i - h / 2) ** 2 


def tilde_x(i: int):
    if i == 0:
        return h ** 2 / (2 * h)
    elif i == N:
        return (radius ** 2 - (radius - h) ** 2) / (2 * h)
    else:
        return ((h * (i + 1)) ** 2 - (h * (i - 1)) ** 2) / (4 * h)


def phi_i(i: int, y: np.array):
    if i == N + 1:
        return (1 - σ) * τ / h * β_2 * h * N * y[N] - τ / h * μ_2 * h * N \
            - 1 / 2 * tilde_x(N) * y[N] + (1 - σ) * τ / 2 * tilde_x(N) * q * y[N] \
            + (1 - σ) * τ / h ** 2 * p_i(N) * (y[N] - y[N - 1])
    else:
        return - tilde_x(i - 1) * y[i - 1] \
            - τ * (1 - σ) / h ** 2 * (p_i(i) * (y[i] - y[i-1]) - p_i(i - 1) * (y[i - 1] - y[i - 2])) \
            + τ * (1 - σ) * tilde_x(i - 1) * q * y[i - 1]


def d_i(i: int):
    return σ * τ / h ** 2 * p_i(i - 1)


def b_i(i: int):
    return σ * τ / h ** 2 * p_i(i)


def c_i(i: int):
    if i == N + 1:
        return - σ * τ / h * β_2 * h * N - 1 / 2 * h * N \
            - σ * τ / 2 * tilde_x(N) * q - d_i(N + 1)
    else:
        return - tilde_x(i - 1) * (1 + τ * σ * q) - (d_i(i) + b_i(i))


print(f"α * τ / h^2 < 1/2: {λ / (c * ρ) * τ / h ** 2:.7f}")


def count_one_layer(v: np.array):
    A_[0, 0] = 1
    A_[0, 1] = -1
    b_[0] = 0
    for i in range(1, N):
        b_[i] = phi_i(i + 1, v)
        A_[i, i - 1] = d_i(i + 1)
        A_[i, i] = c_i(i + 1)
        A_[i, i + 1] = b_i(i + 1)
    b_[N] = phi_i(N + 1, v)
    A_[N, N - 1] = d_i(N + 1) 
    A_[N, N] = c_i(N + 1)
    return np.linalg.solve(A_, b_)


last_times = [M / 4, M / 3, M / 2, M]
times_to_print = [0, 20, 50, M / 2, M]


def count_temperature():
    v = np.ones(N + 1) * initial_temperature
    for j in range(M):
        if j % 100 == 0: 
            plt.plot(np.linspace(0, radius, radius / h + 1), v)
        v = count_one_layer(v)
    print(f'y_0 = {v[0]}')
    print(f'y_N = {v[N]}')
    plt.plot(np.linspace(0, radius, radius / h + 1), v)
    plt.xlabel('radius')
    plt.ylabel('temperature')
    plt.grid()
    plt.title(f'Temperature in {end_time} seconds')
    plt.show()
    return v


if __name__ == '__main__':
    count_temperature()
