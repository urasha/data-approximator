import sys
import math
import numpy as np
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
        self.X = np.array(xs);
        self.Y = np.array(ys)

        N = len(self.X)
        if N < 7 or N > 12:
            QMessageBox.warning(self, 'Ошибка', 'Должно быть от 7 до 12 точек!')
            return

        results = {}

        c1, phi1, S1, R21, r1 = linear_approx(self.X, self.Y)
        results['Linear'] = (c1, phi1, S1, R21, r1)

        c2, phi2, S2, R22 = poly_approx(self.X, self.Y, 2)
        results['Polynomial2'] = (c2, phi2, S2, R22, None)
        c3, phi3, S3, R23 = poly_approx(self.X, self.Y, 3)
        results['Polynomial3'] = (c3, phi3, S3, R23, None)

        ce, phie, Se, R2e = exponential_approx(self.X, self.Y)
        results['Exponential'] = (ce, phie, Se, R2e, None)

        cl, phil, Sl, R2l = log_approx(self.X, self.Y)
        results['Logarithmic'] = (cl, phil, Sl, R2l, None)

        cp, phip, Sp, R2p = power_approx(self.X, self.Y)
        results['Power'] = (cp, phip, Sp, R2p, None)

        best = min(results.items(), key=lambda kv: kv[1][2])[0]

        # Вывод
        self.resultsText.clear()
        for key, label in FUNC_TYPES:
            coeff, phi, S, R2, r = results[key]
            coeff_str = ', '.join(f"{c:.4g}" for c in coeff)
            self.resultsText.append(f"{label}:\nКоэффициенты=({coeff_str}); S={S:.4g}; R²={R2:.4g}\n")
            if key == 'Linear':
                self.resultsText.append(f"Коэффициент корреляции Пирсона r = {r:.4g}\n")
        self.resultsText.append(f"\nЛучшая аппроксимация: {dict(FUNC_TYPES)[best]}")
        R2b = results[best][3]
        self.resultsText.append(f"R² лучшей = {R2b:.4g}")
        if R2b > 0.9:
            self.resultsText.append("Аппроксимация очень хорошая.")
        elif R2b > 0.7:
            self.resultsText.append("Аппроксимация хорошая.")
        else:
            self.resultsText.append("Аппроксимация слабая.")

        # График
        try:
            self.canvas.axes.clear()
            x_list = list(self.X)
            y_list = list(self.Y)
            self.canvas.axes.scatter(x_list, y_list, label='Данные')

            x_min, x_max = min(x_list), max(x_list)
            dx = (x_max - x_min) * 0.05 if x_max != x_min else 1
            x_vals = [x_min + i * (x_max - x_min) / 199 for i in range(200)]
            self.canvas.axes.set_xlim(x_min - dx, x_max + dx)

            for key, label in FUNC_TYPES:
                coeff, _, _, _, _ = results[key]
                y_vals = []
                if key == 'Linear':
                    a, b = coeff
                    y_vals = [a * x + b for x in x_vals]
                elif key in ('Polynomial2', 'Polynomial3'):
                    for x in x_vals:
                        val = 0
                        for j, c in enumerate(coeff):
                            val += c * (x ** j)
                        y_vals.append(val)
                elif key == 'Exponential':
                    a, b = coeff
                    y_vals = [a * math.exp(b * x) for x in x_vals]
                elif key == 'Logarithmic':
                    a, b = coeff
                    y_vals = [a + b * math.log(x) if x > 0 else None for x in x_vals]
                else:
                    a, b = coeff
                    y_vals = [a * (x ** b) for x in x_vals]

                self.canvas.axes.plot(x_vals, y_vals, label=label)

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
