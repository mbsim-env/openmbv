/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#ifndef _OPENMBVGUI_QTRIPPLESLIDER_H_
#define _OPENMBVGUI_QTRIPPLESLIDER_H_

#include <QtGui/QSplitter>
#include <QtGui/QSlider>

namespace OpenMBVGUI {

/*! A vertical triple slider
 * This slider has three handles, where the middle handle can only change between the
 * bounds of the lower and the upper handle.
 * Hence their is a total min/max between the lower and upper handle can change and
 * these handles define a current min/max between the middle handel can change.
 */
class QTripleSlider : public QSplitter {
  Q_OBJECT

  public:

    // constructor
    QTripleSlider(QWidget *parent=NULL);

    // total range getter/setter
    public slots: // declare the setters a slots
    void setTotalRange(int min_, int max_);
    void setTotalMinimum(int min) { setTotalRange(min, totalMax); }
    void setTotalMaximum(int max) { setTotalRange(totalMin, max); }
    public: // the getters may not be slots
    int totalMinimum() { return totalMin; }
    int totalMaximum() { return totalMax; }

    // current range getter/setter
    public slots: // declare the setters a slots
    void setCurrentRange(int min_, int max_);
    void setCurrentMinimum(int min) { setCurrentRange(min, slider->maximum()); }
    void setCurrentMaximum(int max) { setCurrentRange(slider->minimum(), max); }
    public: // the getters may not be slots
    int currentMinimum() { return slider->minimum(); }
    int currentMaximum() { return slider->maximum(); }

    // value getter/setter
    int value() { return slider->value(); }
    void setValue(int value) { return slider->setValue(value); }

  signals:

    // signals on changes
    void currentRangeChanged(int min, int max);
    void sliderMoved(int value);

  protected:

    void resizeEvent(QResizeEvent *event);
    void syncCurrentRangeToSplitterPosition();

    QSlider *slider;
    int totalMin, totalMax;

  protected slots:

    void syncSplitterPositionToCurrentRange();
    void sliderMovedSlot(int value);
};

}

#endif
