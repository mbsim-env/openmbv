import PySide2.QtWidgets



# returns True if this function was called while a Qt-GUI was running (e.g. mbsimgui)
def isGUI():
  if isGUI.value is None:
    isGUI.value = PySide2.QtWidgets.QApplication.instance() is not None
  return isGUI.value
isGUI.value=None



# a helper class for a GUI dialog with matplotlib child widgets
class MatplotlibDialog(PySide2.QtWidgets.QDialog):
  def __init__(self, parent=None):
    import matplotlib
    import PySide2.QtCore
    import os
    super().__init__(parent)
    self.setWindowFlag(PySide2.QtCore.Qt.WindowMaximizeButtonHint, True)
    # enforce matplotlib to use PySide2
    os.environ["QT_API"]="PySide2"
    matplotlib.use('Qt5Agg')
    self.plot = {}
    self.plotToolbar = {}
    self.setModal(True)
  def getPlotWidget(self, id):
    import matplotlib
    import matplotlib.backends.backend_qtagg
    if id not in self.plot:
      self.plot[id] = matplotlib.backends.backend_qtagg.FigureCanvas()
    return self.plot[id]
  def getPlotWidgetToolbar(self, id):
    import matplotlib
    import matplotlib.backends.backend_qtagg
    if id not in self.plotToolbar:
      self.plotToolbar[id] = matplotlib.backends.backend_qtagg.NavigationToolbar2QT(self.getPlotWidget(id), self)
    return self.plotToolbar[id]



def showDialog(D, args=(), maximized=True):
  if isGUI():
    import PySide2.QtCore
    # create and exec the dialog
    d=D(*args)
    if maximized:
      d.setWindowState(PySide2.QtCore.Qt.WindowMaximized)
    d.exec()
  else:
    import PySide2.QtCore
    # create QApplication if noe exists yet
    if PySide2.QtWidgets.QApplication.instance() is None:
      PySide2.QtWidgets.QApplication()
    # create and show the dialog
    d=D(*args)
    if maximized:
      d.setWindowState(PySide2.QtCore.Qt.WindowMaximized)
    d.show()
    # execute QApplication
    PySide2.QtWidgets.QApplication.instance().exec_()
