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



# Shows a instance of the dialog class D. The instance is constructed using args.
# If no QApplication is running yet it is started automatically.
# That is why we need to pass a dialog class D and not a dialog instance d as input
# since it is not possible to create any QWidget instance before a QApplilcation is running.
# For our convinience you mazimized can be used to maximize the created dialog or not.
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
    # create QApplication it it does exists yet: this is needed before any QWidget instance can be created
    if PySide2.QtWidgets.QApplication.instance() is None:
      PySide2.QtWidgets.QApplication()
    # create and show the dialog
    d=D(*args)
    if maximized:
      d.setWindowState(PySide2.QtCore.Qt.WindowMaximized)
    d.show()
    # execute QApplication
    PySide2.QtWidgets.QApplication.instance().exec_()
