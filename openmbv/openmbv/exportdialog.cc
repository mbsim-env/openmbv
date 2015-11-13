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

#include "config.h"
#include "exportdialog.h"
#include <QtGui/QFileDialog>
#include "mainwindow.h"

namespace OpenMBVGUI {

ExportDialog::ExportDialog(QWidget *parent, bool sequence) : QDialog(parent) {
  int row=-1;
  setWindowTitle("Export current frame as PNG");
  setLayout(&dialogLO);
  scaleL.setText("Resolution factor:");
  row++;
  dialogLO.addWidget(&scaleL, row, 0);
  scale.setRange(0.01, 100);
  scale.setSingleStep(0.1);
  scale.setValue(1);
  dialogLO.addWidget(&scale, row, 1, 1, 2);
  backgroundL.setText("Background:");
  row++;
  dialogLO.addWidget(&backgroundL, row, 0);
  transparentRB.setText("Transparent");
  transparentRB.setChecked(true);
  dialogLO.addWidget(&transparentRB, row, 1, 1, 2);
  colorRB.setText("Use scene color");
  row++;
  dialogLO.addWidget(&colorRB, row, 1);
  if(sequence) {
    setWindowTitle("Export frame sequence as PNG"); // overwrite title
    row++;
    fpsL.setText("Frames per second:");
    dialogLO.addWidget(&fpsL, row, 0);
    fps.setRange(0,100);
    fps.setValue(25);
    fps.setDecimals(1);
    dialogLO.addWidget(&fps, row, 1, 1, 2);
    row++;
    speedL.setText("Speed:");
    dialogLO.addWidget(&speedL, row, 0);
    speedLText.setText("use current speed");
    dialogLO.addWidget(&speedLText, row, 1, 1, 2);
    row++;
    frameRangeL.setText("Frame range:");
    dialogLO.addWidget(&frameRangeL, row, 0);
    frameRangeLText.setText("use current range");
    dialogLO.addWidget(&frameRangeLText, row, 1, 1, 2);
  }
  row++;
  fileNameL.setText("File:");
  dialogLO.addWidget(&fileNameL, row, 0);
  fileName.setText("openmbv.png");
  dialogLO.addWidget(&fileName, row, 1);
  fileNameButton.setText("Browse...");
  connect(&fileNameButton, SIGNAL(clicked(bool)), this, SLOT(fileBrowser()));
  dialogLO.addWidget(&fileNameButton, row, 2);
  abort.setText("Abort");
  connect(&abort, SIGNAL(clicked(bool)), this, SLOT(reject()));
  row++;
  dialogLO.addWidget(&abort, row, 0);
  ok.setDefault(true);
  ok.setText("OK");
  connect(&ok, SIGNAL(clicked(bool)), this, SLOT(accept()));
  dialogLO.addWidget(&ok, row, 2);
}

void ExportDialog::fileBrowser() {
  QString name=QFileDialog::getSaveFileName(this, "Save to file", fileName.text(), "PNG-image (*.png)");
  if(name.isNull()) return;
  fileName.setText(name);
}

}
