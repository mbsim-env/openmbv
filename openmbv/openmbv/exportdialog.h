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
    QRadioButton pngSeqenceRB, videoRB;
    QLabel scaleL, backgroundL, fileNameL, speedL, speedLText, fpsL, frameRangeL, frameRangeLText, videoCmdL, bitRateL;
    QTextEdit videoCmd;
    QSpinBox bitRate;
  public:
    ExportDialog(QWidget *parent, bool sequence, bool video);
    double getScale() { return scale.value(); }
    bool getTransparent() { return transparentRB.isChecked(); }
    QString getFileName() { return fileName.text(); }
    double getFPS() { return fps.value(); }
    int getBitRate() { return bitRate.value(); }
    QString getVideoCmd() { return videoCmd.toPlainText(); }
};

}

#endif
