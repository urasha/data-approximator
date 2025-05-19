import math
import numpy as np


def linear_approx(x: np.ndarray, y: np.ndarray):
    a, b = np.polyfit(x, y, 1)
    phi = a * x + b
    S = np.sum((phi - y) ** 2)
    S_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - S / S_tot
    r = np.corrcoef(x, y)[0, 1]
    return [a, b], phi, S, R2, r


def poly_approx(x: np.ndarray, y: np.ndarray, deg: int):
    coeffs = np.polyfit(x, y, deg)
    phi = np.polyval(coeffs, x)
    S = np.sum((phi - y) ** 2)
    S_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - S / S_tot
    return coeffs, phi, S, R2


def exponential_approx(x: np.ndarray, y: np.ndarray):
    mask = y > 0
    x_m, y_m = x[mask], np.log(y[mask])
    b, ln_a = np.polyfit(x_m, y_m, 1)
    a = math.exp(ln_a)
    phi = a * np.exp(b * x)
    S = np.sum((phi - y) ** 2)
    S_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - S / S_tot
    return [a, b], phi, S, R2


def log_approx(x: np.ndarray, y: np.ndarray):
    mask = x > 0
    x_m, y_m = np.log(x[mask]), y[mask]
    b, a = np.polyfit(x_m, y_m, 1)
    phi = a + b * np.log(x)
    S = np.sum((phi - y) ** 2)
    S_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - S / S_tot
    return [a, b], phi, S, R2


def power_approx(x: np.ndarray, y: np.ndarray):
    mask = (x > 0) & (y > 0)
    x_m, y_m = np.log(x[mask]), np.log(y[mask])
    b, ln_a = np.polyfit(x_m, y_m, 1)
    a = math.exp(ln_a)
    phi = a * (x ** b)
    S = np.sum((phi - y) ** 2)
    S_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - S / S_tot
    return [a, b], phi, S, R2
