import csv
import sys
from datetime import datetime
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QLocale, Qt, QTimer
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    # Пытаемся подключить плагинный бэкенд (модули из внешних файлов)
    from model_backend import register_common_modules, train_and_predict, available_models, register_module
    import ml_models as _ml_models
    ModelError = _ml_models.ModelError

    # Регистрируем стандартные модули при старте (если доступны в окружении)
    try:
        register_common_modules()
    except Exception:
        pass

    # Регистрируем модели напрямую из текущей папки проекта
    try:
        _repo_dir = Path(__file__).resolve().parent
        for _f in ['RandomForest.py', 'GradientBoosting.py', 'PLS.py', 'GPR.py', 'MLP.py', 'LinearRegression.py']:
            p = _repo_dir / _f
            try:
                if p.exists():
                    register_module(str(p), alias=p.stem)
            except Exception:
                continue
    except Exception:
        pass

    try:
        print('Available model plugins:', available_models())
    except Exception:
        pass
except Exception:
    # Фоллбек на встроенный простой бэкенд
    from ml_models import ModelError, train_and_predict

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


BASE_DIR = Path(__file__).resolve().parent


def ui_path(*names: str) -> str:
    
    for name in names:
        candidate = BASE_DIR / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Не найден UI-файл. Ожидались варианты: {', '.join(names)}")


class CompareModelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(ui_path("dialog_compare_models.ui", "dialog_compare_models(3).ui", "dialog_compare_models(2).ui"), self)
        self.btnCloseCompare.clicked.connect(self.close)
        self.fill_demo_rows()

    def fill_demo_rows(self):
        rows = [
            ("Random Forest", "12.71", "0.42", "0.91", "0.03 c"),
            ("Gradient Boosting", "12.55", "0.47", "0.89", "0.04 c"),
            ("Linear Regression", "11.90", "0.71", "0.78", "0.01 c"),
        ]
        self.tableCompare.setRowCount(max(self.tableCompare.rowCount(), len(rows)))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.tableCompare.setItem(r, c, QTableWidgetItem(value))
        self.tableCompare.resizeColumnsToContents()


class OptimizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(ui_path("dialog_optimization.ui", "dialog_optimization(3).ui", "dialog_optimization(2).ui"), self)
        self.parent_window = parent
        self.btnRunOptimization.clicked.connect(self.run_optimization)

    def run_optimization(self):
        """Минимальная рабочая заглушка оптимизации: не ломает старый интерфейс и даёт отклик кнопке."""
        self.btnRunOptimization.setText("Готово")
        if self.parent_window is not None:
            self.parent_window.set_forecast_status(
                "Предупреждение",
                "оптимизация пока подключена как демонстрационный модуль",
                "#ff9900",
            )


class MainWindow(QMainWindow):
    PREVIEW_ROWS = 5

    FORECAST_FIELDS = [
        "editFlotationTime",
        "editImpellerFrequency",
        "editAirFlow",
        "editCollectorFlow",
        "editFrotherFlow",
    ]

    FIELD_TITLES = {
        "editFlotationTime": "Время флотации",
        "editImpellerFrequency": "Частота вращения импеллера",
        "editAirFlow": "Расход воздуха",
        "editCollectorFlow": "Расход собирателя",
        "editFrotherFlow": "Расход пенообразователя",
    }

    FIELD_KEYWORDS = {
        "editFlotationTime": ["время", "time", "flotation", "флотац", "флотации"],
        "editImpellerFrequency": ["частота", "frequency", "impeller", "вращ", "импелл"],
        "editAirFlow": ["воздух", "air", "расход воздуха", "airflow"],
        "editCollectorFlow": ["собират", "collector", "собиратель"],
        "editFrotherFlow": ["пенообраз", "frother", "вспенив", "вспен"],
    }

    PARAMETER_WIDGETS_TO_ENABLE = FORECAST_FIELDS + [
        "labelCollector",
        "labelFrother",
        "unitCollector",
        "unitFrother",
    ]

    def __init__(self):
        super().__init__()
        uic.loadUi(ui_path("digital_technologist.ui", "digital_technologist(4).ui", "digital_technologist(3).ui"), self)

        self.forecast_df = None
        self.training_df = None
        self.target_column = None
        self.feature_columns = []
        self.column_select_mode = "target"
        self.last_prediction_result = None

        self._set_numeric_validators()
        self._setup_forecast_tab_old_design()
        self._build_training_tab()
        self._prepare_history_table()
        self._connect_history_buttons()

        # Заполняем список алгоритмов в комбобоксе (`comboModel`) на основе доступных плагинов
        try:
            if hasattr(self, 'comboModel'):
                items = ["Ансамбль моделей (по ум.)"]
                try:
                    # available_models импортируется через model_backend, если он подключён
                    plugin_names = available_models()
                except Exception:
                    plugin_names = []

                def human_name(key: str) -> str:
                    # Если ключ полностью в верхнем регистре (PLS, GPR, MLP), оставляем как есть
                    if str(key).isupper():
                        return str(key)
                    # Разделяем camel case / подчеркивания в более читаемую форму
                    import re
                    s = re.sub(r'(_|-)+', ' ', key)
                    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
                    return s.replace('  ', ' ').strip().title()

                for k in plugin_names:
                    items.append(human_name(k))

                # Если нет зарегистрированных плагинов — добавим стандартные опции
                if len(items) == 1:
                    items.extend(["Random Forest", "Gradient Boosting", "Linear Regression"])

                self.comboModel.clear()
                self.comboModel.addItems(items)
        except Exception:
            pass

        self.set_forecast_status("Готово", "заполните параметры и нажмите «Рассчитать прогноз»", "#3b6fb6")


    def _setup_forecast_tab_old_design(self):

        self.btnCalculate.clicked.connect(self.calculate_forecast)
        self.btnCompareModels.clicked.connect(self.open_compare_dialog)
        self.btnOptimization.clicked.connect(self.open_optimization_dialog)
        # Включаем кнопку оптимизации позже — после расчёта
        self.btnOptimization.setEnabled(False)
        # Включаем кнопку расчёта на всякий случай
        self.btnCalculate.setEnabled(True)
        # Активируем все поля прогноза и связанные метки/юниты
        try:
            self._enable_parameter_fields()
        except Exception:
            # на случай, если некоторые виджеты отсутствуют в UI
            for field_name in self.FORECAST_FIELDS:
                widget = getattr(self, field_name, None)
                if widget is not None:
                    try:
                        widget.setEnabled(True)
                    except Exception:
                        pass
            for extra in ["labelCollector", "labelFrother", "unitCollector", "unitFrother"]:
                w = getattr(self, extra, None)
                if w is not None:
                    try:
                        w.setEnabled(True)
                    except Exception:
                        pass

        # Следим за изменениями в полях ввода и сбрасываем предыдущий результат
        for field_name in self.FORECAST_FIELDS:
            widget = getattr(self, field_name, None)
            if widget is not None and hasattr(widget, 'textChanged'):
                widget.textChanged.connect(self.on_forecast_input_changed)

    def on_forecast_input_changed(self, _value):
        if hasattr(self, 'resultValue'):
            self.resultValue.setText('—')
        self.set_forecast_status('Изменение параметров', 'Нажмите «Рассчитать прогноз» для обновления результата.', '#3b6fb6')

    def _hide_forecast_data_controls_keep_space(self):
        
        group_bg = "#d6d6d6"

        if hasattr(self, "labelData"):
            self.labelData.setFixedHeight(max(self.labelData.sizeHint().height(), 22))
            self.labelData.setText("")
            self.labelData.setStyleSheet(f"background:{group_bg}; color:{group_bg}; border:0px;")
            self.labelData.setEnabled(False)

        for button_name in ["btnLoadExperiment", "btnLoadTraining"]:
            button = getattr(self, button_name, None)
            if button is None:
                continue
            button.setFixedHeight(max(button.sizeHint().height(), 34))
            button.setText("")
            button.setEnabled(False)
            button.setStyleSheet(
                f"background:{group_bg}; color:{group_bg}; border:0px; "
                "text-align:left; padding-left:12px;"
            )

    def _build_training_tab(self):
        
        self.tabTraining = QWidget()
        main_layout = QVBoxLayout(self.tabTraining)
        main_layout.setContentsMargins(24, 18, 24, 18)
        main_layout.setSpacing(12)

        title = QLabel("Обучение модели")
        title.setStyleSheet('font: 13pt "Segoe UI"; font-weight: bold;')
        main_layout.addWidget(title)

        info = QLabel(
            "Загрузите таблицу .xlsx/.csv. Для обучения выберите целевую переменную и параметры "
            "кликом по заголовкам столбцов в таблице."
        )
        info.setWordWrap(True)
        main_layout.addWidget(info)

        data_box = QGroupBox("Данные")
        data_layout = QVBoxLayout(data_box)

        self.btnTrainingLoadForecast = QPushButton("Загрузить экспериментальные данные для прогноза (.xlsx/.csv)")
        self.btnTrainingLoadForecast.setStyleSheet(
            "background:#76a7d3; color:white; border:0px; text-align:left; padding-left:12px; min-height:30px;"
        )
        self.btnTrainingLoadForecast.clicked.connect(self.load_forecast_file)
        data_layout.addWidget(self.btnTrainingLoadForecast)

        self.btnTrainingLoadModel = QPushButton("Загрузить файл для обучения модели (.xlsx/.csv)")
        self.btnTrainingLoadModel.setStyleSheet(
            "background:#e8eef8; color:#333333; border:0px; text-align:left; padding-left:12px; min-height:30px;"
        )
        self.btnTrainingLoadModel.clicked.connect(self.load_training_file)
        data_layout.addWidget(self.btnTrainingLoadModel)

        main_layout.addWidget(data_box)

        controls_layout = QHBoxLayout()
        self.btnChooseTarget = QPushButton("Выбрать целевую переменную")
        self.btnChooseFeatures = QPushButton("Выбрать параметры")
        self.btnClearColumns = QPushButton("Очистить выбор")
        self.btnSaveTrainingSetup = QPushButton("Сохранить выбор")

        self.btnChooseTarget.clicked.connect(lambda: self.set_column_mode("target"))
        self.btnChooseFeatures.clicked.connect(lambda: self.set_column_mode("features"))
        self.btnClearColumns.clicked.connect(self.clear_column_selection)
        self.btnSaveTrainingSetup.clicked.connect(self.save_training_setup)

        for button in [
            self.btnChooseTarget,
            self.btnChooseFeatures,
            self.btnClearColumns,
            self.btnSaveTrainingSetup,
        ]:
            controls_layout.addWidget(button)
        main_layout.addLayout(controls_layout)

        selection_layout = QHBoxLayout()

        target_box = QGroupBox("Целевая переменная")
        target_layout = QVBoxLayout(target_box)
        self.lblTargetColumn = QLabel("Не выбрано")
        self.lblTargetColumn.setWordWrap(True)
        target_layout.addWidget(self.lblTargetColumn)

        features_box = QGroupBox("Параметры для прогнозирования")
        features_layout = QVBoxLayout(features_box)
        self.listFeatureColumns = QListWidget()
        features_layout.addWidget(self.listFeatureColumns)

        selection_layout.addWidget(target_box, 1)
        selection_layout.addWidget(features_box, 2)
        main_layout.addLayout(selection_layout)

        preview_box = QGroupBox("Предпросмотр загруженной таблицы")
        preview_layout = QVBoxLayout(preview_box)
        preview_hint = QLabel(
            "Показываются первые и последние 5 строк. Все столбцы доступны через горизонтальную прокрутку."
        )
        preview_hint.setWordWrap(True)
        preview_layout.addWidget(preview_hint)

        self.tableTrainingPreview = QTableWidget()
        self._setup_table(self.tableTrainingPreview)
        self.tableTrainingPreview.horizontalHeader().sectionClicked.connect(self.handle_column_header_clicked)
        preview_layout.addWidget(self.tableTrainingPreview)
        main_layout.addWidget(preview_box, 1)

        self.lblTrainingStatus = QLabel("Файл не загружен")
        self.lblTrainingStatus.setStyleSheet("background:#f2f2f2; padding:6px;")
        self.lblTrainingStatus.setWordWrap(True)
        main_layout.addWidget(self.lblTrainingStatus)

        self.tabWidget.insertTab(1, self.tabTraining, "Обучение и данные")

    def _enable_parameter_fields(self):
        for widget_name in self.PARAMETER_WIDGETS_TO_ENABLE:
            widget = getattr(self, widget_name, None)
            if widget is not None:
                try:
                    widget.setEnabled(True)
                except Exception:
                    pass
                try:
                    widget.setReadOnly(False)
                except Exception:
                    pass

    def _set_numeric_validators(self):
        validator = QDoubleValidator(self)
        validator.setBottom(0.0)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setLocale(QLocale(QLocale.Russian, QLocale.Russia))

        for field_name in self.FORECAST_FIELDS:
            field = getattr(self, field_name, None)
            if field is not None:
                field.setValidator(validator)

    def _setup_table(self, table: QTableWidget):
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectColumns)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)


    def ask_table_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл с данными",
            "",
            "Таблицы (*.xlsx *.xls *.csv);;Excel (*.xlsx *.xls);;CSV (*.csv);;Все файлы (*)",
        )
        return path

    def read_table_file(self, file_path):
        if pd is None:
            raise RuntimeError("Для загрузки таблиц установите pandas и openpyxl: pip install pandas openpyxl")

        suffix = Path(file_path).suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        if suffix == ".csv":
            return self._read_csv_with_fallback(file_path)
        raise ValueError("Поддерживаются только .xlsx, .xls и .csv")

    def _read_csv_with_fallback(self, file_path):
        last_error = None
        for encoding in ["utf-8-sig", "utf-8", "cp1251"]:
            try:
                # Сначала пробуем явный разделитель точка с запятой — часто встречается в CSV из Excel
                try:
                    return pd.read_csv(file_path, sep=';', engine="python", encoding=encoding)
                except Exception:
                    # Если не сработало, пробуем автоопределение разделителя
                    return pd.read_csv(file_path, sep=None, engine="python", encoding=encoding)
            except Exception as exc:
                last_error = exc
        raise last_error

    def validate_dataframe(self, df):
        if df is None or df.empty:
            return False, "Файл пустой."
        if len(df.columns) < 2:
            return False, "В файле должно быть минимум два столбца."
        if df.dropna(how="all").empty:
            return False, "В файле нет строк с данными."
        return True, "Файл корректный."

    def _validate_or_raise(self, df):
        is_valid, message = self.validate_dataframe(df)
        if not is_valid:
            raise ValueError(message)

    def load_forecast_file(self):
        file_path = self.ask_table_file()
        if not file_path:
            return

        try:
            df = self.read_table_file(file_path)
            self._validate_or_raise(df)
        except Exception as exc:
            self.show_error(f"Не удалось загрузить экспериментальные данные: {exc}")
            return

        self.forecast_df = df
        self.fill_table_preview(self.tableTrainingPreview, df)

        filled = self.fill_forecast_fields_from_row(df.iloc[0])
        self.resultValue.setText("—")
        self.set_training_status(
            f"Экспериментальные данные загружены: {Path(file_path).name}. "
            f"Строк: {len(df)}, столбцов: {len(df.columns)}. Автозаполнено полей прогноза: {filled}."
        )
        self.set_forecast_status("Файл загружен", f"автозаполнено полей: {filled}", "#3bb54a")

    def load_training_file(self):
        file_path = self.ask_table_file()
        if not file_path:
            return

        try:
            df = self.read_table_file(file_path)
            self._validate_or_raise(df)
        except Exception as exc:
            self.show_error(f"Не удалось загрузить файл для обучения: {exc}")
            return

        self.training_df = df
        self.clear_column_selection(update_status=False)
        self.fill_table_preview(self.tableTrainingPreview, df)
        self.set_column_mode("target")
        self.set_training_status(
            f"Файл для обучения загружен: {Path(file_path).name}. "
            f"Строк: {len(df)}, столбцов: {len(df.columns)}. Выберите целевую переменную кликом по заголовку таблицы."
        )


    def make_preview_df(self, df):
        if df is None or df.empty:
            return df
        if len(df) <= self.PREVIEW_ROWS * 2:
            return df.copy()
        return pd.concat([df.head(self.PREVIEW_ROWS), df.tail(self.PREVIEW_ROWS)])

    def fill_table_preview(self, table: QTableWidget, df):
        table.clear()

        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return

        preview = self.make_preview_df(df)
        table.setRowCount(len(preview))
        table.setColumnCount(len(preview.columns))
        table.setHorizontalHeaderLabels([str(col) for col in preview.columns])
        table.setVerticalHeaderLabels([str(idx) for idx in preview.index])

        for row_index, (_, row) in enumerate(preview.iterrows()):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem("" if pd.isna(value) else str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row_index, column_index, item)

        table.resizeRowsToContents()
        table.resizeColumnsToContents()


    def set_column_mode(self, mode):
        self.column_select_mode = mode
        if mode == "target":
            self.set_training_status("Режим выбора: целевая переменная. Кликните по заголовку нужного столбца.")
        else:
            self.set_training_status("Режим выбора параметров. Кликайте по заголовкам столбцов, чтобы добавить или убрать их.")

    def handle_column_header_clicked(self, column_index):
        if self.training_df is None:
            self.set_training_status("Сначала загрузите файл для обучения модели.")
            return

        column_name = str(self.training_df.columns[column_index])
        if self.column_select_mode == "target":
            self._select_target_column(column_name)
        else:
            self._toggle_feature_column(column_name)

    def _select_target_column(self, column_name):
        self.target_column = column_name
        self.feature_columns = [col for col in self.feature_columns if col != column_name]
        self.lblTargetColumn.setText(column_name)
        self.update_feature_list()
        self.set_training_status(f"Выбрана целевая переменная: {column_name}")

    def _toggle_feature_column(self, column_name):
        if column_name == self.target_column:
            self.set_training_status("Этот столбец уже выбран как целевая переменная. Он не может быть параметром.")
            return

        if column_name in self.feature_columns:
            self.feature_columns.remove(column_name)
            self.set_training_status(f"Параметр удалён: {column_name}")
        else:
            self.feature_columns.append(column_name)
            self.set_training_status(f"Параметр добавлен: {column_name}")
        self.update_feature_list()

    def update_feature_list(self):
        self.listFeatureColumns.clear()
        self.listFeatureColumns.addItems(self.feature_columns)

    def clear_column_selection(self, update_status=True):
        self.target_column = None
        self.feature_columns = []
        self.lblTargetColumn.setText("Не выбрано")
        self.listFeatureColumns.clear()
        if update_status:
            self.set_training_status("Выбор столбцов очищен.")

    def save_training_setup(self):
        is_valid, message = self.validate_training_selection()
        if not is_valid:
            self.set_training_status(message)
            return

        self.set_training_status(
            "Выбор сохранён. "
            f"Целевая переменная: {self.target_column}. Параметров выбрано: {len(self.feature_columns)}. "
            "При расчёте прогноза будет использована модель, выбранная в списке алгоритмов."
        )

    def validate_training_selection(self):
        if self.training_df is None:
            return False, "Сначала загрузите файл для обучения модели."
        if not self.target_column:
            return False, "Не выбрана целевая переменная."
        if not self.feature_columns:
            return False, "Не выбраны параметры для прогнозирования."
        return self.validate_numeric_training_columns()

    def validate_numeric_training_columns(self):
        selected_columns = [self.target_column] + self.feature_columns
        data = self.training_df[selected_columns].copy()

        for column in selected_columns:
            data[column] = (
                data[column]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace("\u00a0", "", regex=False)
                .str.strip()
            )
            data[column] = pd.to_numeric(data[column], errors="coerce")

        cleaned = data.dropna()

        if cleaned.empty:
            return False, "В выбранных столбцах нет числовых строк после очистки пропусков."

        if len(cleaned) < 3:
            return False, "Слишком мало числовых строк для обучения модели. Нужно минимум 3."

        return True, f"Данные корректны. Числовых строк после очистки: {len(cleaned)}."

    def calculate_forecast(self):
        values = self.collect_forecast_values()
        if not values:
            self.resultValue.setText("—")
            self.set_forecast_status("Ошибка", "заполните хотя бы один параметр числом", "#d9534f")
            return

        prediction, source = self.predict_concentrate(values)
        if prediction is None:
            self.resultValue.setText("—")
            self.set_forecast_status("Ошибка", source, "#d9534f")
            return

        self.resultValue.setText(f"{prediction:.2f}")
        self.btnOptimization.setEnabled(True)
        self.add_history_row(values, prediction, self.comboModel.currentText())

        if source == "training_model":
            model_name = (
                self.last_prediction_result.model_name
                if self.last_prediction_result is not None
                else self.comboModel.currentText()
            )
            self.set_forecast_status("Успех", f"прогноз рассчитан моделью: {model_name}", "#3bb54a")

    def collect_forecast_values(self):
        values = {}
        for field_name in self.FORECAST_FIELDS:
            widget = getattr(self, field_name, None)
            if widget is None:
                continue

            text = widget.text().strip().replace(",", ".")
            if not text:
                continue

            try:
                values[field_name] = float(text)
            except ValueError:
                self.set_forecast_status("Ошибка", f"поле «{self.FIELD_TITLES[field_name]}» должно быть числом", "#d9534f")
                return {}

        return values

    def predict_concentrate(self, values):
        if self.training_df is None:
            self.last_prediction_result = None
            return None, "Сначала загрузите файл с данными для обучения и выберите целевую переменную и параметры."

        is_valid, message = self.validate_training_selection()
        if not is_valid:
            self.last_prediction_result = None
            return None, message

        try:
            input_values = self.build_feature_vector_for_prediction(values)



            

            self.last_prediction_result = train_and_predict(
                df=self.training_df,
                target_column=self.target_column,
                feature_columns=self.feature_columns,
                input_values=input_values,
                model_name=self.comboModel.currentText(),
            )

            return self.last_prediction_result.prediction, "training_model"

        except ModelError as exc:
            self.last_prediction_result = None
            return None, str(exc)

        except Exception as exc:
            self.last_prediction_result = None
            return None, f"не удалось выполнить расчёт модели: {exc}"

    def build_feature_vector_for_prediction(self, values):
        field_order = [field for field in self.FORECAST_FIELDS if field in values]
        used_fields = set()
        vector = []

        for column in self.feature_columns:
            field = self.find_field_for_column(column, values, used_fields)
            if field is None:
                # если название столбца не совпало по ключевым словам, берём параметры по порядку
                remaining = [f for f in field_order if f not in used_fields]
                if not remaining:
                    raise ValueError(f"для столбца «{column}» нет введённого параметра")
                field = remaining[0]
            print(f"DEBUG MAPPING: column={column}, field={field}, value={values[field]}")
            vector.append(values[field])
            used_fields.add(field)

        return vector

    def find_field_for_column(self, column_name, values, used_fields):
        lower_name = str(column_name).lower()
        for field_name, keywords in self.FIELD_KEYWORDS.items():
            if field_name in values and field_name not in used_fields:
                if any(keyword in lower_name for keyword in keywords):
                    return field_name
        return None

    def fill_forecast_fields_from_row(self, row):
        used_columns = set()
        filled = 0
        lower_columns = {str(col).lower(): col for col in row.index}

        for widget_name, keywords in self.FIELD_KEYWORDS.items():
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue

            source_col = self._find_matching_column(lower_columns, used_columns, keywords)
            if source_col is not None and pd.notna(row[source_col]):
                widget.setText(str(row[source_col]))
                used_columns.add(source_col)
                filled += 1

        if filled == 0:
            filled = self._fill_forecast_fields_by_numeric_order(row)

        return filled

    def _find_matching_column(self, lower_columns, used_columns, keywords):
        for lower_name, original_name in lower_columns.items():
            if original_name in used_columns:
                continue
            if any(keyword in lower_name for keyword in keywords):
                return original_name
        return None

    def _fill_forecast_fields_by_numeric_order(self, row):
        numeric_values = []
        for value in row.values:
            try:
                if pd.notna(value):
                    float(str(value).replace(",", "."))
                    numeric_values.append(value)
            except ValueError:
                continue

        filled = 0
        widgets = [getattr(self, field_name, None) for field_name in self.FORECAST_FIELDS]
        for widget, value in zip([w for w in widgets if w is not None], numeric_values):
            widget.setText(str(value))
            filled += 1
        return filled


    def _prepare_history_table(self):
        if not hasattr(self, "tableHistory"):
            return
        self.tableHistory.setRowCount(0)
        self.tableHistory.resizeColumnsToContents()

    def _connect_history_buttons(self):
        if hasattr(self, "searchEdit"):
            self.searchEdit.textChanged.connect(self.filter_history_rows)
        if hasattr(self, "btnDeleteHistory"):
            self.btnDeleteHistory.clicked.connect(self.delete_selected_history_row)
        if hasattr(self, "btnOpenHistory"):
            self.btnOpenHistory.clicked.connect(self.open_selected_history_row)
        if hasattr(self, "btnExportCsv"):
            self.btnExportCsv.clicked.connect(self.export_history_csv)

    def add_history_row(self, values, prediction, model_name):
        if not hasattr(self, "tableHistory"):
            return

        params = ", ".join(
            f"{self.FIELD_TITLES[field]}: {value:g}"
            for field, value in values.items()
        )
        row_values = [
            datetime.now().strftime("%d.%m.%Y, %H:%M"),
            params,
            f"{prediction:.2f}",
            model_name,
        ]

        self.tableHistory.insertRow(0)
        for column, value in enumerate(row_values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.tableHistory.setItem(0, column, item)
        self.tableHistory.resizeColumnsToContents()

    def filter_history_rows(self, text):
        if not hasattr(self, "tableHistory"):
            return
        text = text.lower().strip()
        for row in range(self.tableHistory.rowCount()):
            row_text = " ".join(
                self.tableHistory.item(row, column).text().lower()
                for column in range(self.tableHistory.columnCount())
                if self.tableHistory.item(row, column) is not None
            )
            self.tableHistory.setRowHidden(row, text not in row_text)

    def delete_selected_history_row(self):
        row = self.tableHistory.currentRow()
        if row < 0:
            self.show_error("Выберите строку истории для удаления.")
            return
        self.tableHistory.removeRow(row)

    def open_selected_history_row(self):
        row = self.tableHistory.currentRow()
        if row < 0:
            self.show_error("Выберите строку истории для просмотра.")
            return
        values = []
        for column in range(self.tableHistory.columnCount()):
            header = self.tableHistory.horizontalHeaderItem(column).text()
            item = self.tableHistory.item(row, column)
            values.append(f"{header}: {item.text() if item is not None else ''}")
        QMessageBox.information(self, "Запись истории", "\n".join(values))

    def export_history_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить историю расчётов",
            "history.csv",
            "CSV (*.csv);;Все файлы (*)",
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow([
                    self.tableHistory.horizontalHeaderItem(column).text()
                    for column in range(self.tableHistory.columnCount())
                ])
                for row in range(self.tableHistory.rowCount()):
                    writer.writerow([
                        self.tableHistory.item(row, column).text()
                        if self.tableHistory.item(row, column) is not None else ""
                        for column in range(self.tableHistory.columnCount())
                    ])
            self.set_forecast_status("Успех", "история экспортирована в CSV", "#3bb54a")
        except Exception as exc:
            self.show_error(f"Не удалось экспортировать историю: {exc}")


    def open_compare_dialog(self):
        dialog = CompareModelsDialog(self)
        dialog.exec_()

    def open_optimization_dialog(self):
        dialog = OptimizationDialog(self)
        dialog.exec_()

    def set_training_status(self, text):
        if hasattr(self, "lblTrainingStatus"):
            self.lblTrainingStatus.setText(text)

    def set_forecast_status(self, title, text, color):
        if hasattr(self, "statusMessage"):
            self.statusMessage.setText(f'<span style="color:{color};">{title}</span>: {text}')

    def show_error(self, text):
        QMessageBox.critical(self, "Ошибка", text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
