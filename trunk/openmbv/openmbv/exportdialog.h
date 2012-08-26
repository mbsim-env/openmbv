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
    QDoubleSpinBox scale, fps;
    QPushButton fileNameButton, ok, abort;
    QLineEdit fileName;
    QGridLayout dialogLO;
    QRadioButton transparentRB, colorRB;
    QLabel scaleL, backgroundL, fileNameL, speedL, speedLText, fpsL, frameRangeL, frameRangeLText;
  public:
    ExportDialog(QWidget *parent, bool sequence);
    double getScale() { return scale.value(); }
    bool getTransparent() { return transparentRB.isChecked(); }
    QString getFileName() { return fileName.text(); }
    double getFPS() { return fps.value(); }
  protected slots:
    void fileBrowser();
};

#endif
