import sys
import math
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QFileDialog,
    QLabel, QMessageBox
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

        self.X = np.array([])
        self.Y = np.array([])
        self.results = {}
        self.initUI()

    def initUI(self):
        main = QWidget()
        root = QHBoxLayout()

        # Левая панель
        left = QVBoxLayout()
        left.addWidget(QLabel('Введите данные (минимум 7 и максимум 12 строк), формат: x y'))
        self.inputText = QTextEdit()
        self.inputText.setPlaceholderText('Пример:\n1.0 2.1\n2.0 3.5\n...')
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
        self.textRes = QTextEdit()
        self.textRes.setReadOnly(True)
        left.addWidget(self.textRes)

        # Правая панель: график
        right = QVBoxLayout()
        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        right.addWidget(self.canvas)

        root.addLayout(left, 3)
        root.addLayout(right, 5)
        main.setLayout(root)
        self.setCentralWidget(main)

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'Текстовые файлы (*.txt *.csv)')
        if not fname:
            return
        try:
            data = np.loadtxt(fname)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError('Неверный формат файла')
            lines = [f"{row[0]} {row[1]}" for row in data]
            self.inputText.setPlainText("\n".join(lines))
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось загрузить: {e}')

    def parse_input(self):
        lines = self.inputText.toPlainText().splitlines()
        xs, ys = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0].replace(',', '.')))
                    ys.append(float(parts[1].replace(',', '.')))
                except:
                    pass
        return np.array(xs), np.array(ys)

    def calculate(self):
        self.X, self.Y = self.parse_input()
        N = len(self.X)
        if N < 7 or N > 12:
            QMessageBox.warning(self, 'Ошибка', 'Требуется от 7 до 12 точек!')
            return
        self.results.clear()
        S_tot = np.sum((self.Y - np.mean(self.Y)) ** 2)

        # Линейная
        a1, b1 = np.polyfit(self.X, self.Y, 1)
        phi1 = a1 * self.X + b1
        S1 = np.sum((phi1 - self.Y) ** 2)
        r = np.corrcoef(self.X, self.Y)[0, 1]
        R2_1 = 1 - S1 / S_tot
        self.results['Linear'] = dict(coeff=[a1, b1], phi=phi1, S=S1, r=r, R2=R2_1)

        # Полином 2-го порядка
        p2 = np.polyfit(self.X, self.Y, 2)
        phi2 = np.polyval(p2, self.X)
        S2 = np.sum((phi2 - self.Y) ** 2)
        self.results['Polynomial2'] = dict(coeff=p2, phi=phi2, S=S2, R2=1 - S2 / S_tot)

        # Полином 3-го порядка
        p3 = np.polyfit(self.X, self.Y, 3)
        phi3 = np.polyval(p3, self.X)
        S3 = np.sum((phi3 - self.Y) ** 2)
        self.results['Polynomial3'] = dict(coeff=p3, phi=phi3, S=S3, R2=1 - S3 / S_tot)

        # Экспоненциальная y=a·e^(b·x)
        mask = self.Y > 0
        xa, ya = self.X[mask], np.log(self.Y[mask])
        ba, la = np.polyfit(xa, ya, 1)
        a_exp = math.exp(la)
        phi_exp = a_exp * np.exp(ba * self.X)
        Se = np.sum((phi_exp - self.Y) ** 2)
        self.results['Exponential'] = dict(coeff=[a_exp, ba], phi=phi_exp, S=Se, R2=1 - Se / S_tot)

        # Логарифмическая y=a + b·ln(x)
        mask2 = self.X > 0
        xb, yb = np.log(self.X[mask2]), self.Y[mask2]
        bb, a_log = np.polyfit(xb, yb, 1)
        phi_log = a_log + bb * np.log(self.X)
        Sl = np.sum((phi_log - self.Y) ** 2)
        self.results['Logarithmic'] = dict(coeff=[a_log, bb], phi=phi_log, S=Sl, R2=1 - Sl / S_tot)

        # Степенная y=a·x^b
        xc, yc = np.log(self.X[mask2]), np.log(self.Y[mask2])
        bc, la2 = np.polyfit(xc, yc, 1)
        a_pow = math.exp(la2)
        phi_pow = a_pow * (self.X ** bc)
        Sp = np.sum((phi_pow - self.Y) ** 2)
        self.results['Power'] = dict(coeff=[a_pow, bc], phi=phi_pow, S=Sp, R2=1 - Sp / S_tot)

        # Определение лучшей аппроксимации по минимальному S
        best = min(self.results.items(), key=lambda kv: kv[1]['S'])[0]

        # Вывод результатов
        self.textRes.clear()
        for key, rus in FUNC_TYPES:
            rct = self.results[key]
            coeff_str = ', '.join(f"{c:.4g}" for c in rct['coeff'])
            self.textRes.append(f"{rus}:\nКоэффициенты ({coeff_str}); S = {rct['S']:.4g}; R² = {rct['R2']:.4g}\n")
            if key == 'Linear':
                self.textRes.append(f"Корреляция Пирсона r = {rct['r']:.4g}\n")
        self.textRes.append(f"\nЛучшая аппроксимация: {dict(FUNC_TYPES)[best]}")
        R2_best = self.results[best]['R2']
        self.textRes.append(f"R² лучшей = {R2_best:.4g}")
        if R2_best > 0.9:
            self.textRes.append("Аппроксимация очень хорошая.")
        elif R2_best > 0.7:
            self.textRes.append("Аппроксимация хорошая.")
        else:
            self.textRes.append("Аппроксимация слабая.")

        # Построение графиков
        self.canvas.axes.clear()
        self.canvas.axes.scatter(self.X, self.Y, label='Данные')
        xline = np.linspace(min(self.X), max(self.X), 200)
        dx = (max(self.X) - min(self.X)) * 0.05
        self.canvas.axes.set_xlim(min(self.X) - dx, max(self.X) + dx)
        for key, rus in FUNC_TYPES:
            rct = self.results[key]
            if key == 'Linear':
                yline = rct['coeff'][0] * xline + rct['coeff'][1]
            elif key.startswith('Polynomial'):
                yline = np.polyval(rct['coeff'], xline)
            elif key == 'Exponential':
                a, b = rct['coeff']
                yline = a * np.exp(b * xline)
            elif key == 'Logarithmic':
                a, b = rct['coeff']
                yline = a + b * np.log(xline)
            else:
                a, b = rct['coeff']
                yline = a * (xline ** b)
            self.canvas.axes.plot(xline, yline, label=rus)
        self.canvas.axes.legend()
        self.canvas.axes.grid(True)
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
