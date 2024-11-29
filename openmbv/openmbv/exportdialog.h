/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef _OPENMBVGUI_EXPORTDIALOG_H_
#define _OPENMBVGUI_EXPORTDIALOG_H_

#include <QDialog>
#include <QPushButton>
#include <QCheckBox>
#include <QButtonGroup>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QGridLayout>
#include <QRadioButton>
#include <QTextEdit>

namespace OpenMBVGUI {

class ExportDialog : public QDialog {
  Q_OBJECT
  protected:
    QDoubleSpinBox scale, fps;
    QPushButton fileNameButton, ok, abort;
    QLineEdit fileName;
    QGridLayout dialogLO;
    QLabel variantL;
    QButtonGroup colorBG, variantBG;
    QRadioButton transparentRB, colorRB;
    QRadioButton pngSeqenceRB;
    QLabel scaleL, backgroundL, fileNameL, speedL, speedLText, fpsL, frameRangeL, frameRangeLText, bitRateL, skipL, keepL;
    QCheckBox skip, keep;
    QSpinBox bitRate;
    QString outputFileExt;
  public:
    ExportDialog(QWidget *parent, bool sequence, bool video);
    double getScale() const { return scale.value(); }
    bool getTransparent() const { return transparentRB.isChecked(); }
    QString getFileName() const;
    double getFPS() const { return fps.value(); }
    int getBitRate() const { return bitRate.value(); }
    bool keepPNGs() const { return keep.isChecked(); }
    bool skipPNGGeneration() const { return skip.isChecked(); }
};

}

#endif
