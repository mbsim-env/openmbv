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
