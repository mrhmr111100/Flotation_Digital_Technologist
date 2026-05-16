import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QTableWidgetItem


class CompareModelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("dialog_compare_models.ui", self)
        self.btnCloseCompare.clicked.connect(self.close)
        rows = [
            ("Random Forest", "12.71", "0.42", "0.91", "0.03 c"),
            ("Gradient Boosting", "12.55", "0.47", "0.89", "0.04 c"),
            ("Linear Regression", "11.90", "0.71", "0.78", "0.01 c"),
        ]
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.tableCompare.setItem(r, c, QTableWidgetItem(value))


class OptimizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("dialog_optimization.ui", self)
        self.btnRunOptimization.clicked.connect(self.run_optimization)

    def run_optimization(self):
        # алгоритм оптимизации (пока просто имитация)
        self.btnRunOptimization.setText("Готово")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("digital_technologist.ui", self)

        self.btnCalculate.clicked.connect(self.calculate_forecast)
        self.btnCompareModels.clicked.connect(self.open_compare_dialog)
        self.btnOptimization.clicked.connect(self.open_optimization_dialog)

        self.prepare_history_table()

    def prepare_history_table(self):
        rows = [
            (
                "06.05.2026, 14:20",
                "n мин, n Гц, n %, n г/т, n г/т",
                "12.71",
                "Random Forest",
            ),
            (
                "06.05.2026, 13:40",
                "n мин, n Гц, n %, n г/т, n г/т",
                "12.55",
                "Ансамбль моделей",
            ),
        ]
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.tableHistory.setItem(r, c, QTableWidgetItem(value))
        self.tableHistory.resizeColumnsToContents()

    def calculate_forecast(self):
        # просто пример. здесь будет запуск модели и расчет прогноза
        self.resultValue.setText("12.71")
        self.statusMessage.setText(
            '<span style="color:#3bb54a;">Успех</span>: прогноз рассчитан'
        )
        self.btnOptimization.setEnabled(True)

    def open_compare_dialog(self):
        dialog = CompareModelsDialog(self)
        dialog.exec_()

    def open_optimization_dialog(self):
        dialog = OptimizationDialog(self)
        dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
