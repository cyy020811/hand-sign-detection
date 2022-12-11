import sys

from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QPushButton


class resultDialog(QMessageBox):
    def __init__(self):
        super().__init__()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Press me for a dialog!")
        button.clicked.connect(self.result)
        self.setCentralWidget(button)

    def button_clicked(self, s):
        self.dlg = QMessageBox(self)
        self.dlg.setWindowTitle("I have a question!")
        self.dlg.setText("This is a simple dialog")
        button = self.dlg.exec()

        if button == QMessageBox.StandardButton.Ok:
            print("OK!")

    def result(self):
        print('LPM: ' + str(60*60/(60-61)))
        self.rdlg = resultDialog()
        self.rdlg.setWindowTitle("Result")
        self.rdlg.setText("Your result of " + 'hard' + ' is:\n\nLPM: ' + str(60*60/(60-61)))
        self.rdlg.addButton(QMessageBox.StandardButton.Close)
        button = self.rdlg.exec()

        if button == QMessageBox.StandardButton.Close:
            print("Close!")

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()