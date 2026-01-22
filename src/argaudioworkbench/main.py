import sys

from PySide6 import QtWidgets

from argaudioworkbench.ui.main_window import MainWindow


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ArgAudioWorkbench")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
