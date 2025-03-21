"""MBXMLUtils preprocessor helper functions specific for Qt based applications"""

import os
import matplotlib
import PySide2.QtWidgets



def isGUI():
  """returns True if this function was called while a Qt-GUI was running (e.g. mbsimgui)"""
  return _isGUI

# initialize the Qt module
if PySide2.QtWidgets.QApplication.instance() is None:
  # if a QApplication is not already instantiated then we are not running a GUI program (mbsimgui) and start a QAppliation now
  _isGUI=False
  PySide2.QtWidgets.QApplication()
else:
  # if a QApplication is already instantiated then we are running a GUI program (mbsimgui) and need not to start a QApplication
  _isGUI=True

# enforce matplotlib to use PySide2
os.environ["QT_API"]="PySide2"
matplotlib.use('Qt5Agg')



class MatplotlibDialog(PySide2.QtWidgets.QDialog):
  """A helper class for a GUI dialog with matplotlib child widgets."""
  def __init__(self, parent=None):
    """Construct a MatplotlibDialog instance and pass parent to QDialog"""
    super().__init__(parent)
    import PySide2.QtCore
    self.setWindowFlag(PySide2.QtCore.Qt.WindowMaximizeButtonHint, True)
    self.plot = {}
    self.plotToolbar = {}
    self.setModal(True)
  def createPlotWidget(self, id, **fig_kwargs):
    """Create a matplotlib widget named "id". Throws if such a widget already exists
    fig_kwargs is passed to matplotlib.figure.Figure(...) when creating the corresponding figure"""
    import matplotlib.backends.backend_qtagg
    import matplotlib.figure
    if id in self.plot:
      raise RuntimeError(f"A plot widget with id={id} already exists.")
    self.plot[id] = matplotlib.backends.backend_qtagg.FigureCanvas(matplotlib.figure.Figure(**fig_kwargs))
    return self.plot[id]
  def getPlotWidget(self, id):
    """Get, or create and get, a matplotlib widget named "id".
    The returned widget has a "figure" attribute being the matplotlib figure
    which itself has the matplotlib function "subplots" to create axes"""
    if id not in self.plot:
      self.createPlotWidget(id)
    return self.plot[id]
  def getPlotWidgetToolbar(self, id):
    """Get, or create and get, the matplotlib navigation toolbar widget for the matplotlib widget named "id", see getPlotWidget"""
    import matplotlib.backends.backend_qtagg
    if id not in self.plotToolbar:
      self.plotToolbar[id] = matplotlib.backends.backend_qtagg.NavigationToolbar2QT(self.getPlotWidget(id), self)
    return self.plotToolbar[id]



class StdMatplotlibDialog(MatplotlibDialog):
  """This class create a dialog with the standard matplotlib layout of a matplotlib figure:
  the toolbar at the top of the dialog and a single main figure widget.
  getFigure can be used to get the matplotlib figure e.g. to create subplots
  The dialog is then shown modal using execWidget."""
  def __init__(self, parent=None, **fig_kwargs):
    super().__init__(parent)
    layout = PySide2.QtWidgets.QVBoxLayout()
    self.setLayout(layout)
    self.stdPlot=self.createPlotWidget("plot", **fig_kwargs)
    layout.addWidget(self.getPlotWidgetToolbar("plot"))
    layout.addWidget(self.stdPlot)
  def getFigure(self):
    return self.stdPlot.figure



def execWidget(w, maximized=True):
  """Use this function to show (execute) the main widget for a model specific UI (e.g. a plot window).
  You should use this function instead of w.exec() (or w.show()) to support showing widgets in GUI and none GUI programs.
  For none GUI program this function will automatically create a Qt event loop until the Widget is open.
  The widget is shown modal.
  "maximized" can be used for your convinience to show the widget maximized (calls w.setWindowState(PySide2.QtCore.Qt.WindowMaximized))."""
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
