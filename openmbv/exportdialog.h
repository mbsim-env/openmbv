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

#ifndef _EXPORTDIALOG_H_
#define _EXPORTDIALOG_H_

#include "config.h"
#include <QtGui/QDialog>
#include <QtGui/QPushButton>
#include <QtGui/QLineEdit>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QLabel>
#include <QtGui/QGridLayout>
#include <QtGui/QRadioButton>

class ExportDialog : public QDialog {
  Q_OBJECT
  protected:
    QDoubleSpinBox scale, speed, fps;
    QSpinBox startFrame, endFrame;
    QPushButton colorButton, fileNameButton, ok, abort;
    QLineEdit fileName;
    QColor color;
    QGridLayout dialogLO;
    QRadioButton transparentRB, colorRB;
    QLabel scaleL, backgroundL, fileNameL, speedL, fpsL, frameRangeL;
  public:
    ExportDialog(QWidget *parent, bool sequence);
    double getScale() { return scale.value(); }
    bool getTransparent() { return transparentRB.isChecked(); }
    QColor getColor() { return color; }
    QString getFileName() { return fileName.text(); }
    double getFPS() { return fps.value(); }
    double getSpeed() { return speed.value(); }
    int getStartFrame() { return startFrame.value(); }
    int getEndFrame() { return endFrame.value(); }
  protected slots:
    void colorToggled(bool enabled);
    void colorButtonClicked();
    void fileBrowser();
};

#endif
