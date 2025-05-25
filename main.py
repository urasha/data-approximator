import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QFileDialog,
    QLabel, QMessageBox, QHeaderView, QTableWidget, QTableWidgetItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from approx_funcs import (
    linear_approx, poly_approx, exponential_approx,
    log_approx, power_approx
)

FUNC_TYPES = [
    ('Linear', 'Линейная y = a·x + b'),
    ('Polynomial2', 'Полином 2-го порядка y = a·x² + b·x + c'),
    ('Polynomial3', 'Полином 3-го порядка y = a·x³ + b·x² + c·x + d'),
    ('Exponential', 'Экспоненциальная y = a·e^(b·x)'),
    ('Logarithmic', 'Логарифмическая y = a + b·ln(x)'),
    ('Power', 'Степенная y = a·x^b')
]


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Аппроксимация данных")
        self.setGeometry(100, 100, 1200, 700)
        self.X = []
        self.Y = []
        self.initUI()

    def initUI(self):
        main = QWidget()
        hl = QHBoxLayout()

        left = QVBoxLayout()
        left.addWidget(QLabel('Введите данные (минимум 7 и максимум 12):'))
        self.inputText = QTextEdit()
        self.inputText.setPlaceholderText('x y\nпример:\n1.0 2.1\n2.0 3.5\n...')
        left.addWidget(self.inputText)

        btns = QHBoxLayout()
        self.loadBtn = QPushButton('Загрузить из файла')
        self.loadBtn.clicked.connect(self.load_file)
        btns.addWidget(self.loadBtn)
        self.calcBtn = QPushButton('Выполнить расчет')
        self.calcBtn.clicked.connect(self.calculate)
        btns.addWidget(self.calcBtn)
        left.addLayout(btns)

        left.addWidget(QLabel('Результаты:'))
        self.resultsText = QTextEdit()
        self.resultsText.setReadOnly(True)
        left.addWidget(self.resultsText)

        right = QVBoxLayout()
        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        right.addWidget(self.canvas)

        hl.addLayout(left, 3)
        hl.addLayout(right, 5)
        main.setLayout(hl)
        self.setCentralWidget(main)

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Открыть файл данных', '', 'Текстовые файлы (*.txt *.csv)')
        if fname:
            try:
                data = [line.strip().split() for line in open(fname) if line.strip()]
                lines = [f"{x} {y}" for x, y in data]
                self.inputText.setPlainText("\n".join(lines))
            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Ошибка загрузки: {e}')

    def calculate(self):
        # Разбор ввода
        xs, ys = [], []
        for line in self.inputText.toPlainText().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0].replace(',', '.')))
                    ys.append(float(parts[1].replace(',', '.')))
                except:
                    pass
        self.X = xs
        self.Y = ys
        N = len(xs)
        if N < 7 or N > 12:
            QMessageBox.warning(self, 'Ошибка', 'Должно быть от 7 до 12 точек!')
            return

        temp = {}
        try:
            c, phi, S, R2, r = linear_approx(xs, ys)
            temp['Linear'] = (c, phi, S, R2, r)
        except:
            temp['Linear'] = None

        for deg, key in [(2, 'Polynomial2'), (3, 'Polynomial3')]:
            try:
                c, phi, S, R2 = poly_approx(xs, ys, deg)
                temp[key] = (c, phi, S, R2, None)
            except:
                temp[key] = None

        if any(y > 0 for y in ys):
            try:
                c, phi, S, R2 = exponential_approx(xs, ys)
                temp['Exponential'] = (c, phi, S, R2, None)
            except:
                temp['Exponential'] = None
        else:
            temp['Exponential'] = None

        if any(x > 0 for x in xs):
            try:
                c, phi, S, R2 = log_approx(xs, ys)
                temp['Logarithmic'] = (c, phi, S, R2, None)
            except:
                temp['Logarithmic'] = None
        else:
            temp['Logarithmic'] = None

        if any(x > 0 and y > 0 for x, y in zip(xs, ys)):
            try:
                c, phi, S, R2 = power_approx(xs, ys)
                temp['Power'] = (c, phi, S, R2, None)
            except:
                temp['Power'] = None
        else:
            temp['Power'] = None

        results = {}
        for k, v in temp.items():
            if v is not None:
                coeff, phi, S, R2, r = v
                sigma = math.sqrt(S / N)
                results[k] = (coeff, phi, S, R2, r, sigma)

        if not results:
            QMessageBox.critical(self, 'Ошибка', 'Ни одна модель не применима!')
            return

        best = min(results.items(), key=lambda kv: kv[1][5])[0]

        self.resultsText.clear()
        for key, label in FUNC_TYPES:
            if key not in results:
                self.resultsText.append(f"{label}: неприменимо")
                continue
            coeff, phi, S, R2, r, sigma = results[key]
            coeff_str = ', '.join(f"{c:.4g}" for c in coeff)
            self.resultsText.append(
                f"{label}:\n  Коэффициенты = ({coeff_str})\n  σ = {sigma:.9g}, R² = {R2:.4g}"
            )
            if key == 'Linear':
                self.resultsText.append(f"  r (Пирсон) = {r:.4g}")
        self.resultsText.append(f"\nЛучшая по σ: {dict(FUNC_TYPES)[best]}")

        try:
            self.canvas.axes.clear()
            self.canvas.axes.scatter(xs, ys, label='Данные')
            x_min, x_max = min(xs), max(xs)
            dx = (x_max - x_min) * 0.05 if x_max != x_min else 1
            x_vals = [x_min + i * (x_max - x_min) / 199 for i in range(200)]
            self.canvas.axes.set_xlim(x_min - dx, x_max + dx)

            for key, label in FUNC_TYPES:
                if key not in results:
                    continue
                coeff = results[key][0]
                if key == 'Linear':
                    a, b = coeff
                    y_vals = [a * x + b for x in x_vals]
                elif key in ('Polynomial2', 'Polynomial3'):
                    y_vals = []
                    for x in x_vals:
                        v = sum(coeff[j] * (x ** j) for j in range(len(coeff)))
                        y_vals.append(v)
                elif key == 'Exponential':
                    a, b = coeff
                    y_vals = [a * math.exp(b * x) for x in x_vals]
                elif key == 'Logarithmic':
                    a, b = coeff
                    y_vals = [a + b * math.log(x) if x > 0 else None for x in x_vals]
                else:
                    a, b = coeff
                    y_vals = [a * (x ** b) for x in x_vals]

                pts = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None]
                if pts:
                    xs_p, ys_p = zip(*pts)
                    self.canvas.axes.plot(xs_p, ys_p, label=label)

            self.canvas.axes.legend()
            self.canvas.axes.grid(True)
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка графика', str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
