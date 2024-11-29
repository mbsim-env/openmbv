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

#include "config.h"
#include "exportdialog.h"
#include <QFileDialog>
#include "utils.h"
#include "mainwindow.h"

namespace OpenMBVGUI {

bool keepValue = false;
bool skipValue = false;

ExportDialog::ExportDialog(QWidget *parent, bool sequence, bool video) : QDialog(parent) {

  int row=-1;
  setWindowTitle("Export current frame as PNG");
  setLayout(&dialogLO);
  if(video) {
    row++;
    QString tt2("If enabled the generated PNG sequence files are kept (not delete). This can be used to avoid regeneration of the sequence at a further run if e.g. only the bitrate should be changed.");
    keepL.setText("Keep PNG sequence files:");
    keepL.setToolTip(tt2);
    dialogLO.addWidget(&keepL, row, 0);
    keep.setChecked(keepValue);
    keep.setToolTip(tt2);
    connect(&keep, &QCheckBox::clicked, [](bool checked) {
      keepValue=checked;
    });
    dialogLO.addWidget(&keep, row, 1, 1, 2);
    row++;
    QString tt("If enabled a existing set of PNG files form a previous run is used. If disabled a new sequence is generated first. This can be used to avoid regeneration of the sequence if e.g. only the bitrate should be changed.");
    skipL.setText("Skip PNG generation:");
    skipL.setToolTip(tt);
    dialogLO.addWidget(&skipL, row, 0);
    connect(&skip, &QCheckBox::clicked, [this](bool checked) {
      skipValue=checked;
      scaleL.setDisabled(checked);
      scale.setDisabled(checked);
      backgroundL.setDisabled(checked);
      transparentRB.setDisabled(checked);
      colorRB.setDisabled(checked);
      speedL.setDisabled(checked);
      speedLText.setDisabled(checked);
      frameRangeL.setDisabled(checked);
      frameRangeLText.setDisabled(checked);
    });
    skip.setChecked(skipValue);
    skip.clicked(skipValue);
    skip.setToolTip(tt);
    dialogLO.addWidget(&skip, row, 1, 1, 2);
  }
  scaleL.setText("Resolution factor:");
  row++;
  dialogLO.addWidget(&scaleL, row, 0);
  scale.setRange(0.01, 100);
  scale.setSingleStep(0.1);
  scale.setValue(appSettings->get<double>(AppSettings::exportdialog_resolutionfactor));
  dialogLO.addWidget(&scale, row, 1, 1, 2);
  backgroundL.setText("Background:");
  row++;
  dialogLO.addWidget(&backgroundL, row, 0);
  colorBG.addButton(&transparentRB);
  transparentRB.setText("Transparent");
  transparentRB.setChecked(!appSettings->get<bool>(AppSettings::exportdialog_usescenecolor));
  dialogLO.addWidget(&transparentRB, row, 1);
  colorBG.addButton(&colorRB);
  colorRB.setText("Use scene color");
  colorRB.setChecked(appSettings->get<bool>(AppSettings::exportdialog_usescenecolor));
  dialogLO.addWidget(&colorRB, row, 2);
  if(sequence) {
    setWindowTitle("Export frame sequence as PNG"); // overwrite title
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
    row++;
    if(!video)
      fpsL.setText("Frames per second:");
    else
      fpsL.setText("Frames per second (%F):");
    dialogLO.addWidget(&fpsL, row, 0);
    fps.setRange(0,100);
    fps.setSingleStep(1);
    fps.setValue(appSettings->get<double>(AppSettings::exportdialog_fps));
    fps.setDecimals(1);
    dialogLO.addWidget(&fps, row, 1, 1, 2);
  }
  if(!video) {
    outputFileExt="png";
    row++;
    QString ttfile;
    if(!sequence) {
      fileNameL.setText("Output PNG filename:");
      ttfile=".png is appended if it does not end so.";
    }
    else {
      fileNameL.setText("Output PNG base-filename:");
      ttfile="For each frame a underscore and a 6-digit sequence number is\n"
         "appended to the base-filename, e.g: openmbv_000341.png.\n"
         ".png is appended if it does not end so.";
    }
    fileNameL.setToolTip(ttfile);
    dialogLO.addWidget(&fileNameL, row, 0);
    fileName.setText(appSettings->get<QString>(AppSettings::exportdialog_filename_png));
    fileName.setToolTip(ttfile);
    dialogLO.addWidget(&fileName, row, 1);
    fileNameButton.setText("Browse...");
    connect(&fileNameButton, &QPushButton::clicked, this, [this](){
      QString name=QFileDialog::getSaveFileName(this, "Output PNG file", fileName.text(), "PNG-image (*.png)");
      if(name.isNull()) return;
      fileName.setText(name);
    });
    fileNameButton.setToolTip(ttfile);
    dialogLO.addWidget(&fileNameButton, row, 2);
  }
  else {
    outputFileExt=appSettings->get<QString>(AppSettings::exportdialog_videoext);
    setWindowTitle("Export frame sequence as video"); // overwrite title
    row++;
    bitRateL.setText("Bit-Rate [kBit/s] (%B):");
    dialogLO.addWidget(&bitRateL, row, 0);
    bitRate.setRange(1,1000000);
    bitRate.setSingleStep(500);
    bitRate.setValue(appSettings->get<int>(AppSettings::exportdialog_bitrate));
    dialogLO.addWidget(&bitRate, row, 1, 1, 2);
    row++;
    QString ttfile("."+outputFileExt+" is appended if it does not end so.");
    fileNameL.setText("Output video filename (%O):");
    fileNameL.setToolTip(ttfile);
    dialogLO.addWidget(&fileNameL, row, 0);
    fileName.setText(appSettings->get<QString>(AppSettings::exportdialog_filename_video));
    fileName.setToolTip(ttfile);
    dialogLO.addWidget(&fileName, row, 1);
    fileNameButton.setText("Browse...");
    connect(&fileNameButton, &QPushButton::clicked, this, [this](){
      QString name=QFileDialog::getSaveFileName(this, "Output video file", fileName.text(), "Video (*."+outputFileExt+")");
      if(name.isNull()) return;
      fileName.setText(name);
    });
    fileNameButton.setToolTip(ttfile);
    dialogLO.addWidget(&fileNameButton, row, 2);
  }
  abort.setText("Abort");
  connect(&abort, &QPushButton::clicked, this, &ExportDialog::reject);
  row++;
  dialogLO.addWidget(&abort, row, 0);
  ok.setDefault(true);
  ok.setText("OK");
  connect(&ok, &QPushButton::clicked, this, [this, sequence, video](){
    appSettings->set(AppSettings::exportdialog_resolutionfactor, scale.value());
    appSettings->set(AppSettings::exportdialog_usescenecolor, colorRB.isChecked());
    if(sequence)
      appSettings->set(AppSettings::exportdialog_fps, fps.value());
    if(!video) {
      appSettings->set(AppSettings::exportdialog_filename_png, fileName.text());
    }
    else {
      appSettings->set(AppSettings::exportdialog_filename_video, fileName.text());
      appSettings->set(AppSettings::exportdialog_bitrate, bitRate.value());
    }
    accept();
  });
  dialogLO.addWidget(&ok, row, 2);
}

QString ExportDialog::getFileName() const {
  //mfmf make absolute
  if(fileName.text().toLower().endsWith("."+outputFileExt.toLower()))
    return QFileInfo(fileName.text()).absoluteFilePath();
  return QFileInfo(fileName.text()+"."+outputFileExt).absoluteFilePath();
}

}
