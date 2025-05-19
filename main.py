import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QFileDialog,
    QTextEdit, QLabel, QMessageBox, QHeaderView
)
from PyQt5.QtCore import Qt

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
        self.setGeometry(100,100,1200,700)

        self.X = []
        self.Y = []
        self.results = {}
        self.initUI()

    def initUI(self):
        main = QWidget()
        hl = QHBoxLayout()

        # левая панель
        left = QVBoxLayout()
        self.table = QTableWidget(12,2)
        self.table.setHorizontalHeaderLabels(['X','Y'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        left.addWidget(QLabel('Данные (минимум 7 точек, максимум 12)'))
        left.addWidget(self.table)

        btns = QHBoxLayout()
        self.loadBtn = QPushButton('Загрузить из файла')
        self.loadBtn.clicked.connect(self.load_file)
        btns.addWidget(self.loadBtn)
        self.calcBtn = QPushButton('Выполнить расчет')
        self.calcBtn.clicked.connect(self.calculate)
        btns.addWidget(self.calcBtn)
        left.addLayout(btns)

        left.addWidget(QLabel('Результаты:'))
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        left.addWidget(self.text)

        # правая панель: график
        right = QVBoxLayout()
        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        right.addWidget(self.canvas)

        hl.addLayout(left,3)
        hl.addLayout(right,5)
        main.setLayout(hl)
        self.setCentralWidget(main)

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Открыть файл','', 'Текстовые файлы (*.txt *.csv)')
        if not fname: return
        try:
            data = np.loadtxt(fname, delimiter=None)
            if data.ndim!=2 or data.shape[1]<2:
                raise ValueError('Неверный формат файла')
            N = data.shape[0]
            self.table.setRowCount(N)
            for i in range(N):
                self.table.setItem(i,0,QTableWidgetItem(str(data[i,0])))
                self.table.setItem(i,1,QTableWidgetItem(str(data[i,1])))
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось загрузить: {e}')

    def get_data(self):
        xs, ys = [],[]
        for r in range(self.table.rowCount()):
            xi = self.table.item(r,0)
            yi = self.table.item(r,1)
            if xi and yi:
                try:
                    xs.append(float(xi.text().replace(',', '.')))
                    ys.append(float(yi.text().replace(',', '.')))
                except: pass
        return np.array(xs), np.array(ys)

    def calculate(self):
        self.X, self.Y = self.get_data()
        N = len(self.X)
        if N < 7 or N > 12:
            QMessageBox.warning(self,'Ошибка','Нужно от 7 до 12 точек для анализа!')
            return
        self.results.clear()
        S_tot = np.sum((self.Y - np.mean(self.Y))**2)

        # Линейная
        a1, b1 = np.polyfit(self.X, self.Y,1)
        phi1 = a1*self.X + b1
        S1 = np.sum((phi1-self.Y)**2)
        r = np.corrcoef(self.X,self.Y)[0,1]
        R2_1 = 1 - S1/S_tot
        self.results['Linear'] = dict(coeff=[a1,b1], phi=phi1, S=S1, r=r, R2=R2_1)

        # Полином 2-го порядка
        p2 = np.polyfit(self.X,self.Y,2)
        phi2 = np.polyval(p2,self.X)
        S2 = np.sum((phi2-self.Y)**2)
        self.results['Polynomial2'] = dict(coeff=p2, phi=phi2, S=S2, R2=1-S2/S_tot)

        # Полином 3-го порядка
        p3 = np.polyfit(self.X,self.Y,3)
        phi3 = np.polyval(p3,self.X)
        S3 = np.sum((phi3-self.Y)**2)
        self.results['Polynomial3'] = dict(coeff=p3, phi=phi3, S=S3, R2=1-S3/S_tot)

        # Экспоненциальная y=a·e^(b·x)
        mask = self.Y>0
        xa, ya = self.X[mask], np.log(self.Y[mask])
        ba, la = np.polyfit(xa,ya,1)
        a_exp = math.exp(la)
        phi_exp = a_exp*np.exp(ba*self.X)
        Se = np.sum((phi_exp-self.Y)**2)
        self.results['Exponential'] = dict(coeff=[a_exp,ba], phi=phi_exp, S=Se, R2=1-Se/S_tot)

        # Логарифмическая y=a + b·ln(x)
        mask2 = self.X>0
        xb, yb = np.log(self.X[mask2]), self.Y[mask2]
        bb, a_log = np.polyfit(xb,yb,1)
        phi_log = a_log + bb*np.log(self.X)
        Sl = np.sum((phi_log-self.Y)**2)
        self.results['Logarithmic'] = dict(coeff=[a_log,bb], phi=phi_log, S=Sl, R2=1-Sl/S_tot)

        # Степенная y=a·x^b
        xc, yc = np.log(self.X[mask2]), np.log(self.Y[mask2])
        bc, la2 = np.polyfit(xc,yc,1)
        a_pow = math.exp(la2)
        phi_pow = a_pow*(self.X**bc)
        Sp = np.sum((phi_pow-self.Y)**2)
        self.results['Power'] = dict(coeff=[a_pow,bc], phi=phi_pow, S=Sp, R2=1-Sp/S_tot)

        # Лучшая по S
        best = min(self.results.items(), key=lambda kv: kv[1]['S'])[0]

        # Вывод результатов
        self.text.clear()
        for key, rus in FUNC_TYPES:
            rct = self.results[key]
            coeff_str = ', '.join(f"{c:.4g}" for c in rct['coeff'])
            self.text.append(f"{rus}:\nКоэффициенты ({coeff_str}); S = {rct['S']:.4g}; R² = {rct['R2']:.4g}\n")
            if key=='Linear':
                self.text.append(f"Корреляция Пирсона r = {rct['r']:.4g}\n")
        self.text.append(f"\nЛучшая аппроксимация: {dict(FUNC_TYPES)[best]}")
        R2_best = self.results[best]['R2']
        self.text.append(f"R² лучшей = {R2_best:.4g}")
        if R2_best > 0.9:
            self.text.append("Аппроксимация очень хорошая.")
        elif R2_best > 0.7:
            self.text.append("Аппроксимация хорошая.")
        else:
            self.text.append("Аппроксимация слабая.")

        # Построение графика
        self.canvas.axes.clear()
        self.canvas.axes.scatter(self.X,self.Y,label='Данные')
        xline = np.linspace(min(self.X), max(self.X),200)
        dx = (max(self.X)-min(self.X))*0.05
        self.canvas.axes.set_xlim(min(self.X)-dx, max(self.X)+dx)
        for key, rus in FUNC_TYPES:
            rct = self.results[key]
            if key=='Linear':
                yline = rct['coeff'][0]*xline + rct['coeff'][1]
            elif key.startswith('Polynomial'):
                yline = np.polyval(rct['coeff'], xline)
            elif key=='Exponential':
                a,b = rct['coeff']; yline = a*np.exp(b*xline)
            elif key=='Logarithmic':
                a,b = rct['coeff']; yline = a + b*np.log(xline)
            else:
                a,b = rct['coeff']; yline = a*(xline**b)
            self.canvas.axes.plot(xline, yline, label=rus)
        self.canvas.axes.legend()
        self.canvas.axes.grid(True)
        self.canvas.draw()

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())