import PySide2.QtWidgets



# returns True if this function was called while a Qt-GUI was running (e.g. mbsimgui)
def isGUI():
  return _isGUI

# initialize the Qt module
if PySide2.QtWidgets.QApplication.instance() is None:
  # if a QApplication is not already instantiated then we are not running a GUI program (mbsimgui) and start a QAppliation now
  _isGUI=False
  PySide2.QtWidgets.QApplication()
else:
  # if a QApplication is already instantiated then we are running a GUI program (mbsimgui) and need not to start a QApplication
  _isGUI=True



# A helper class for a GUI dialog with matplotlib child widgets.
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
  # Get, or create and get, a matplotlib widget named "id".
  # The returned widget has a "figure" attribute being the matplotlib figure
  # which itself has the matplotlib function "subplots" to create axes
  def getPlotWidget(self, id):
    import matplotlib
    import matplotlib.backends.backend_qtagg
    if id not in self.plot:
      self.plot[id] = matplotlib.backends.backend_qtagg.FigureCanvas()
    return self.plot[id]
  # Get, or create and get, the matplotlib navigation toolbar widget for the matplotlib widget named "id", see getPlotWidget
  def getPlotWidgetToolbar(self, id):
    import matplotlib
    import matplotlib.backends.backend_qtagg
    if id not in self.plotToolbar:
      self.plotToolbar[id] = matplotlib.backends.backend_qtagg.NavigationToolbar2QT(self.getPlotWidget(id), self)
    return self.plotToolbar[id]



# This class create a dialog with the standard matplotlib layout of a matplotlib figure:
# the toolbar at the top of the dialog and a single main figure widget.
# getFigure can be used to get the matplotlib figure e.g. to create subplots
# The dialog is then shown modal using execWidget.
class StdMatplotlibDialog(MatplotlibDialog):
  def __init__(self, parent=None):
    super().__init__(parent)
    layout = PySide2.QtWidgets.QVBoxLayout()
    self.setLayout(layout)
    layout.addWidget(self.getPlotWidgetToolbar("plot"))
    self.plot=self.getPlotWidget("plot")
    layout.addWidget(self.plot)
  def getFigure(self):
    return self.plot.figure



# Use this function to show (execute) the main widget for a model specific UI (e.g. a plot window).
# You should use this function instead of w.exec() (or w.show()) to support showing widgets in GUI and none GUI programs.
# For none GUI program this function will automatically create a Qt event loop until the Widget is open.
# The widget is shown modal.
# "maximized" can be used for your convinience to show the widget maximized (calls w.setWindowState(PySide2.QtCore.Qt.WindowMaximized)).
def execWidget(w, maximized=True):
  import PySide2.QtCore
  if maximized:
    w.setWindowState(PySide2.QtCore.Qt.WindowMaximized)
  if isGUI():
    # show the dialog model and wait for it to exit (a global event loop in already running by the GUI program)
    w.exec()
  else:
    # in a one GUI program show the dialog and return and
    w.show()
    # start a local event loop (none is running till now) which waits for the dialog to exit (hence the dialog is modal)
    PySide2.QtWidgets.QApplication.instance().exec_()
