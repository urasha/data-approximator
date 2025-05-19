import math


def _solve_linear_system(a, b):
    n = len(b)
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[max_row] = a[max_row], a[i]
        b[i], b[max_row] = b[max_row], b[i]

        pivot = a[i][i]
        if abs(pivot) < 1e-12:
            continue

        for j in range(i, n):
            a[i][j] /= pivot
        b[i] /= pivot

        for r in range(i + 1, n):
            factor = a[r][i]
            for c in range(i, n):
                a[r][c] -= factor * a[i][c]
            b[r] -= factor * b[i]

    coeff = [0.0] * n
    for i in reversed(range(n)):
        s = b[i] - sum(a[i][j] * coeff[j] for j in range(i + 1, n))
        coeff[i] = s / a[i][i] if abs(a[i][i]) > 1e-12 else 0.0

    return coeff


def linear_approx(x, y):
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(x[i] * y[i] for i in range(n))

    denom = n * sxx - sx * sx
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n

    phi = [a * xi + b for xi in x]

    S = sum((phi[i] - y[i]) ** 2 for i in range(n))
    y_mean = sy / n
    S_tot = sum((yi - y_mean) ** 2 for yi in y)
    R2 = 1 - S / S_tot if S_tot != 0 else 0.0

    cov = sum((x[i] - sx / n) * (y[i] - y_mean) for i in range(n))
    var_x = sum((xi - sx / n) ** 2 for xi in x)
    var_y = sum((yi - y_mean) ** 2 for yi in y)
    r = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0.0

    return [a, b], phi, S, R2, r


def poly_approx(x, y, deg):
    n = len(x)
    m = deg + 1

    A = [[0.0] * m for _ in range(m)]
    B = [0.0] * m
    for i in range(m):
        for j in range(m):
            A[i][j] = sum(x[k] ** (i + j) for k in range(n))
        B[i] = sum(y[k] * x[k] ** i for k in range(n))

    coeffs = _solve_linear_system(A, B)
    phi = [sum(coeffs[j] * xi ** j for j in range(m)) for xi in x]

    S = sum((phi[i] - y[i]) ** 2 for i in range(n))
    y_mean = sum(y) / n
    S_tot = sum((yi - y_mean) ** 2 for yi in y)
    R2 = 1 - S / S_tot if S_tot != 0 else 0.0

    return coeffs, phi, S, R2


def exponential_approx(x, y):
    xs, ys = [], []
    for xi, yi in zip(x, y):
        if yi > 0:
            xs.append(xi)
            ys.append(math.log(yi))

    coeff_lin, _, _, _, _ = linear_approx(xs, ys)
    b, ln_a = coeff_lin
    a = math.exp(ln_a)

    phi = [a * math.exp(b * xi) for xi in x]
    S = sum((phi[i] - y[i]) ** 2 for i in range(len(x)))
    y_mean = sum(y) / len(y)
    S_tot = sum((yi - y_mean) ** 2 for yi in y)
    R2 = 1 - S / S_tot if S_tot != 0 else 0.0

    return [a, b], phi, S, R2


def log_approx(x, y):
    xs, ys = [], []
    for xi, yi in zip(x, y):
        if xi > 0:
            xs.append(math.log(xi))
            ys.append(yi)

    coeff_lin, _, _, _, _ = linear_approx(xs, ys)
    b, a = coeff_lin

    phi = [a + b * math.log(xi) if xi > 0 else float('nan') for xi in x]
    S = sum((phi[i] - y[i]) ** 2 for i in range(len(x)))
    y_mean = sum(y) / len(y)
    S_tot = sum((yi - y_mean) ** 2 for yi in y)
    R2 = 1 - S / S_tot if S_tot != 0 else 0.0

    return [a, b], phi, S, R2


def power_approx(x, y):
    xs, ys = [], []
    for xi, yi in zip(x, y):
        if xi > 0 and yi > 0:
            xs.append(math.log(xi))
            ys.append(math.log(yi))

    coeff_lin, _, _, _, _ = linear_approx(xs, ys)
    b, ln_a = coeff_lin
    a = math.exp(ln_a)

    phi = [a * (xi ** b) for xi in x]
    S = sum((phi[i] - y[i]) ** 2 for i in range(len(x)))
    y_mean = sum(y) / len(y)
    S_tot = sum((yi - y_mean) ** 2 for yi in y)
    R2 = 1 - S / S_tot if S_tot != 0 else 0.0

    return [a, b], phi, S, R2
