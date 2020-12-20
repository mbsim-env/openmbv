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
#include <QFileDialog>
#include <QSettings>
#include "mainwindow.h"

namespace OpenMBVGUI {

ExportDialog::ExportDialog(QWidget *parent, bool sequence, bool video) : QDialog(parent) {
  QSettings settings;

  int row=-1;
  setWindowTitle("Export current frame as PNG");
  setLayout(&dialogLO);
  scaleL.setText("Resolution factor:");
  row++;
  dialogLO.addWidget(&scaleL, row, 0);
  scale.setRange(0.01, 100);
  scale.setSingleStep(0.1);
  scale.setValue(settings.value("exportdialog/resolutionfactor", 1).toDouble());
  dialogLO.addWidget(&scale, row, 1, 1, 2);
  backgroundL.setText("Background:");
  row++;
  dialogLO.addWidget(&backgroundL, row, 0);
  colorBG.addButton(&transparentRB);
  transparentRB.setText("Transparent");
  transparentRB.setChecked(!settings.value("exportdialog/usescenecolor", true).toBool());
  dialogLO.addWidget(&transparentRB, row, 1);
  colorBG.addButton(&colorRB);
  colorRB.setText("Use scene color");
  colorRB.setChecked(settings.value("exportdialog/usescenecolor", true).toBool());
  dialogLO.addWidget(&colorRB, row, 2);
  if(sequence) {
    setWindowTitle("Export frame sequence as PNG"); // overwrite title
    row++;
    fpsL.setText("Frames per second:");
    dialogLO.addWidget(&fpsL, row, 0);
    fps.setRange(0,100);
    fps.setSingleStep(1);
    fps.setValue(settings.value("exportdialog/fps", 25).toDouble());
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
  if(!video) {
    row++;
    fileNameL.setText("Base-PNG-File:");
    QString tt("For each frame a underscore and a 6-digit sequence number is\n"
               "appended to the base-file, e.g: openmbv_000341.png");
    fileNameL.setToolTip(tt);
    dialogLO.addWidget(&fileNameL, row, 0);
    fileName.setText(settings.value("exportdialog/filename/png", "openmbv.png").toString());
    fileName.setToolTip(tt);
    dialogLO.addWidget(&fileName, row, 1);
    fileNameButton.setText("Browse...");
    connect(&fileNameButton, &QPushButton::clicked, [this](){
      QString name=QFileDialog::getSaveFileName(this, "Save PNG sequence to file", fileName.text(), "PNG-image (*.png)");
      if(name.isNull()) return;
      fileName.setText(name);
    });
    fileNameButton.setToolTip(tt);
    dialogLO.addWidget(&fileNameButton, row, 2);
  }
  else {
    setWindowTitle("Export frame sequence as video"); // overwrite title
    row++;
    fileNameL.setText("Video-File:");
    dialogLO.addWidget(&fileNameL, row, 0);
    fileName.setText(settings.value("exportdialog/filename/video", "openmbv.webm").toString());
    dialogLO.addWidget(&fileName, row, 1);
    fileNameButton.setText("Browse...");
    connect(&fileNameButton, &QPushButton::clicked, [this](){
      QString name=QFileDialog::getSaveFileName(this, "Save video to file", fileName.text(), "Video (*.*)");
      if(name.isNull()) return;
      fileName.setText(name);
    });
    dialogLO.addWidget(&fileNameButton, row, 2);
    bitRateL.setText("Bit-Rate [KiBit/s]:");
    row++;
    dialogLO.addWidget(&bitRateL, row, 0);
    bitRate.setRange(1,102400);
    bitRate.setSingleStep(512);
    bitRate.setValue(settings.value("exportdialog/bitrate", "2048").toInt());
    dialogLO.addWidget(&bitRate, row, 1, 1, 2);
    QString tt("<p>Command to generate the video from a sequence of PNG-files:</p>"
               "<ul>"
               "  <li>The input PNG sequence files are in the current directory of this command (being a temporary dir) "
               "      and are named openmbv_000000.png, openmbv_000001.png, ...\n"
               "  <li>The absolute path of the output video file can be accessed using %O\n"
               "  <li>The bit-rate (in unit Bits per second) can be accessed using %B\n"
               "  <li>The frame-rate (a floating point number) can be accessed using %F"
               "</ul>");
    videoCmdL.setText("Video Command");
    videoCmdL.setToolTip(tt);
    row++;
    dialogLO.addWidget(&videoCmdL, row, 0);
    row++;
    videoCmd.setText(settings.value("exportdialog/videocmd",
      "ffmpeg -framerate %F -i openmbv_%06d.png -c:v libvpx-vp9 -b:v %B -pass 1 -f null /dev/null && "
      "ffmpeg -framerate %F -i openmbv_%06d.png -c:v libvpx-vp9 -b:v %B -pass 2 %O").toString());
    videoCmd.setToolTip(tt);
    dialogLO.addWidget(&videoCmd, row, 0, 1, 3);
  }
  abort.setText("Abort");
  connect(&abort, &QPushButton::clicked, this, &ExportDialog::reject);
  row++;
  dialogLO.addWidget(&abort, row, 0);
  ok.setDefault(true);
  ok.setText("OK");
  connect(&ok, &QPushButton::clicked, [this, sequence, video](){
    QSettings settings;
    settings.setValue("exportdialog/resolutionfactor", scale.value());
    settings.setValue("exportdialog/usescenecolor", colorRB.isChecked());
    if(sequence)
      settings.setValue("exportdialog/fps", fps.value());
    if(!video) {
      settings.setValue("exportdialog/filename/png", fileName.text());
    }
    else {
      settings.setValue("exportdialog/filename/video", fileName.text());
      settings.setValue("exportdialog/bitrate", bitRate.value());
      settings.setValue("exportdialog/videocmd", videoCmd.toPlainText());
    }
    accept();
  });
  dialogLO.addWidget(&ok, row, 2);
}

}
