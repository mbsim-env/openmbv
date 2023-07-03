#include "config.h"
#include "QTripleSlider.h"
#include <QFrame>
#include <QResizeEvent>
#include "utils.h"

using namespace std;

namespace OpenMBVGUI {

QTripleSlider::QTripleSlider(QWidget *parent) : QSplitter(Qt::Vertical, parent) {
  QString iconPath(QUrl::fromLocalFile(Utils::getIconPath().c_str()).toLocalFile());
  // the appearance of the splitter handle
  setStyleSheet(QString(
    "QSplitter::handle {"
    "  border-width: 1px;"
    "  border-style: outset;"
    "  border-color: palette(shadow);"
    "  height: 6px;"
    "  background-image: url(%1/knurl.png);"
    "}"
    "QSplitter::handle:hover {"
    "  background-color: palette(highlight);"
    "}"
  ).arg(iconPath));

  // add the Slider and two dummy Frames to the Splitter
  slider=new QSlider(Qt::Vertical);
  auto *topFrame=new QFrame;
  auto *bottomFrame=new QFrame;
  addWidget(topFrame);
  addWidget(slider);
  addWidget(bottomFrame);

  // the appearance of the two Frames
  topFrame->setStyleSheet(QString(
    "QFrame {"
    "  border-width: 1px;"
    "  border-left-style: inset;"
    "  border-right-style: inset;"
    "  border-top-style: inset;"
    "  border-color: palette(shadow);"
    "  background-image: url(%1/cover.png);"
    "  background-position: bottom left;"
    "}"
  ).arg(iconPath));
  bottomFrame->setStyleSheet(QString(
    "QFrame {"
    "  border-width: 1px;"
    "  border-left-style: inset;"
    "  border-right-style: inset;"
    "  border-bottom-style: inset;"
    "  border-color: palette(shadow);"
    "  background-image: url(%1/cover.png);"
    "  background-position: top left;"
    "}"
  ).arg(iconPath));
  topFrame->setMaximumSize(slider->sizeHint().width(), topFrame->maximumSize().height());
  bottomFrame->setMaximumSize(slider->sizeHint().width(), bottomFrame->maximumSize().height());

  // init the top and bottom Frame with zero size and the Slider with maximal size
  QList<int> sizeArray;
  sizeArray.append(0);
  sizeArray.append(1);
  sizeArray.append(0);
  setSizes(sizeArray);

  // do not remove any widget if the corrosponding size becomes 0
  setChildrenCollapsible(false);

  // pipe Slider signals throught
  connect(slider, &QSlider::sliderMoved, this, &QTripleSlider::sliderMovedSlot);

  // connections
  connect(this, &QTripleSlider::splitterMoved, this, &QTripleSlider::syncSplitterPositionToCurrentRange);
}

void QTripleSlider::syncSplitterPositionToCurrentRange() {
  // the current total size (in pt)
  int size=sizes()[0]+sizes()[1]+sizes()[2]-slider->minimumSizeHint().height();
  // the new current min
  double slopeMin=static_cast<double>(totalMax-totalMin)/size;
  int newMin=static_cast<int>(slopeMin*sizes()[2]+totalMin+0.5);
  // the new current max
  double slopeMax=static_cast<double>(totalMin-totalMax)/size;
  int newMax=static_cast<int>(slopeMax*sizes()[0]+totalMax+0.5);

  // do nothing if nothing has changed
  if(newMin==slider->minimum() && newMax==slider->maximum()) return;
  // set new current range and emit
  int oldValue=slider->value(); // slave old value before
  slider->setRange(newMin, newMax);
  currentRangeChanged(newMin, newMax);

  // do nothing if slider value has not changed
  if(oldValue==slider->value()) return;
  // emit 
  sliderMoved(slider->value());
}

void QTripleSlider::setTotalRange(int min_, int max_) {
  bool currentRangeHasChanged=false;
  int oldValue=slider->value();

  // if currentRange equals old totalRange change also currentRange
  if(slider->minimum()==totalMin) { currentRangeHasChanged=true; slider->setMinimum(min_); }
  if(slider->maximum()==totalMax) { currentRangeHasChanged=true; slider->setMaximum(max_); }
  // set new totalRange
  totalMin=min_;
  totalMax=max_;
  // adapt currentRange if out of totalRange
  if(slider->minimum()<totalMin) { currentRangeHasChanged=true; slider->setMinimum(totalMin); }
  if(slider->maximum()>totalMax) { currentRangeHasChanged=true; slider->setMaximum(totalMax); }

  // do nothing if current range has not changed
  if(!currentRangeHasChanged) return;
  // sync current range and emit
  syncCurrentRangeToSplitterPosition();
  currentRangeChanged(slider->minimum(), slider->maximum());

  // do nothing if value has not changed
  if(oldValue==slider->value()) return;
  // emit
  sliderMoved(slider->value());
}

void QTripleSlider::setCurrentRange(int min, int max) {
  int oldValue=slider->value();

  // restrict current range to total range
  if(min<totalMin) min=totalMin;
  if(max>totalMax) max=totalMax;

  // do nothing if value has not changed
  if(min==slider->minimum() && max==slider->maximum()) return;
  // set current range and sync
  slider->setRange(min, max);
  syncCurrentRangeToSplitterPosition();

  // do nothing if value has not changed
  if(oldValue==slider->value()) return;
  // emit
  sliderMoved(slider->value());
}

void QTripleSlider::resizeEvent(QResizeEvent *event) {
  // resize the widget
  QSplitter::resizeEvent(event);

  // update
  syncCurrentRangeToSplitterPosition();
}

void QTripleSlider::syncCurrentRangeToSplitterPosition() {
  // reset the splitter such that the old currentRange matches (inverse calculation of syncSplitterPositionToCurrentRange)

  // the new current total size (in pt)
  int size=sizes()[0]+sizes()[1]+sizes()[2]-slider->minimumSizeHint().height();

  // the new current min
  double slopeMin=static_cast<double>(totalMax-totalMin)/size;
  int s2=slopeMin ? static_cast<int>((slider->minimum()-totalMin)/slopeMin+0.5) : 0;

  // the new current max
  double slopeMax=static_cast<double>(totalMin-totalMax)/size;
  int s0=slopeMax ? static_cast<int>((slider->maximum()-totalMax)/slopeMax+0.5) : 0;

  // set new splitter positions
  QList<int> sizeArray;
  sizeArray.append(s0);
  sizeArray.append(size+slider->minimumSizeHint().height()-s0-s2);
  sizeArray.append(s2);
  setSizes(sizeArray);
}

void QTripleSlider::sliderMovedSlot(int value) {
  sliderMoved(value);
}

}
