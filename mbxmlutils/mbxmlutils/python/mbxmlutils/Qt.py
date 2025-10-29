"""MBXMLUtils preprocessor helper functions specific for Qt based applications"""

import os
import matplotlib
import PySide2.QtWidgets
import PySide2.QtCore



def isGUI():
  """returns True if this function was called while a Qt-GUI was running (e.g. mbsimgui)"""
  return _isGUI

# initialize the Qt module
if PySide2.QtCore.QCoreApplication.instance() is None:
  import sys
  # if a QApplication is not already instantiated then we are not running a GUI program (mbsimgui) and start a QAppliation now
  _isGUI=False
  if sys.platform.startswith("linux"):
    # Linux may be headless -> instantiate QCoreApplication or QApplication
    if os.environ.get("DISPLAY", "")!="" or os.environ.get("WAYLAND_DISPLAY", "")!="":
      PySide2.QtWidgets.QApplication()
    else:
      PySide2.QtCore.QCoreApplication()
  else:
    # In Windows instantiate always QApplication
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



def blockUntilDialoagsAreClosed(*args):
  """This function blocks until (currently open) dialogs listed as arguments are closed."""
  openDialogs=len(args)
  el=PySide2.QtCore.QEventLoop()
  def finished():
    nonlocal openDialogs
    openDialogs-=1
    if openDialogs==0:
      el.exit(0)
  for d in args:
    d.finished.connect(finished)
  el.exec_()



def onArtistClick(artist, func):
  """If artist (e.g. a line, text, ...) is clicked then the function func(artist) is run.
  However, this works only if the default toolbar actions is currently active.
  If no toolbar action is active (the artist is clickable) then also the mouse cursor changes
  to a hand when the mouse is over the artist."""
  artist.set_picker(True)
  def onPick(event):
    if event.artist == artist:
      func()
  artist.figure.canvas.mpl_connect("pick_event", onPick)
  def onMouseMove(event):
    if artist.figure.canvas.widgetlock.locked():
      return
    if artist.contains(event)[0]:
      artist.figure.canvas.setCursor(PySide2.QtCore.Qt.PointingHandCursor)
    else:
      artist.figure.canvas.setCursor(PySide2.QtCore.Qt.ArrowCursor)
  artist.figure.canvas.mpl_connect("motion_notify_event", onMouseMove)



class BitBlitManager:
  """A helper class for efficient interactive matplotlib figures/axes."""
  def __init__(self, figOrAxes, animatedArtists):
    """Creates a bit blit manger with a matplotlib figure or axes object and a list of artists.
    The figure or axes area is then bit blit with the static plot data as background the list of 
    artists as foreground."""
    if isinstance(figOrAxes, matplotlib.figure.Figure):
      self.fig = figOrAxes
    elif isinstance(figOrAxes, matplotlib.axes.Axes):
      self.fig = figOrAxes.figure
    else:
      raise RuntimeError("wrong first argument")
    self.bbox = figOrAxes.bbox
    self.background = None
    self.artists = []
    for a in animatedArtists:
      a.set_animated(True)
      self.artists.append(a)
    self.fig.canvas.mpl_connect("draw_event", self._draw)
  def update(self):
    """Update/redraw the figure or axes.
    Draw the static plot data as background and the the list of artists at foreground."""
    if self.background is None:
      self._draw(None)
    self.fig.canvas.restore_region(self.background)
    self._drawAnimated()
    self.fig.canvas.blit(self.bbox)
  def _draw(self, event):
    self.background = self.fig.canvas.copy_from_bbox(self.bbox)
    self._drawAnimated()
  def _drawAnimated(self):
    for a in self.artists:
      self.fig.draw_artist(a)
