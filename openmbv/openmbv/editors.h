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

#ifndef _OPENMBVGUI_EDITORS_H_
#define _OPENMBVGUI_EDITORS_H_

#include <fmatvec/atom.h>
#pragma push_macro("NDEBUG") // reworkaround some debugging bug in Coin3D
#ifndef NDEBUG
#define NDEBUG
#endif
#include <Inventor/nodes/SoRotation.h>
#pragma pop_macro("NDEBUG")
#include <Inventor/nodes/SoTranslation.h>
#include <openmbvcppinterface/polygonpoint.h>
#include "utils.h"
#include <QtGui/QDialog>
#include <QtGui/QCheckBox>
#include <QtGui/QGridLayout>
#include <QtGui/QVBoxLayout>
#include <QtGui/QScrollArea>
#include <QtGui/QAction>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QComboBox>
#include <QtGui/QLineEdit>
#include <QtGui/QTableWidget>
#include <QtGui/QColorDialog>
#include <Inventor/draggers/SoCenterballDragger.h>
#include <Inventor/draggers/SoDragger.h>
#include <Inventor/nodes/SoSwitch.h>
#include <cmath>
#include <queue>

namespace OpenMBVGUI {

class Editor;
class Object;





class PropertyDialog : public QDialog {
  Q_OBJECT

  friend class TransRotEditor;
  public:
    PropertyDialog(QObject *obj);
    ~PropertyDialog();
    void setParentObject(QObject *obj);
    void addSmallRow(const QIcon& icon, const std::string& name, QWidget *widget);
    void addLargeRow(const QIcon& icon, const std::string& name, QWidget *widget);
    void addSmallRow(const QIcon& icon, const std::string& name, QLayout *subLayout);
    void addLargeRow(const QIcon& icon, const std::string& name, QLayout *subLayout);
    void updateHeader();
    QObject* getParentObject() { return parentObject; }
    void addPropertyAction(QAction *action);
    void addPropertyActionGroup(QActionGroup *actionGroup);
    void addContextAction(QAction *action);
    QMenu *getContextMenu() { return contextMenu; }
    void addEditor(Editor *child);
    QList<QAction*> getActions();
  protected:
    QMenu *contextMenu;
    QObject* parentObject;
    QGridLayout *layout, *mainLayout;
    std::vector<Editor*> editor;
  protected slots:
    void openDialogSlot();
    void deleteObjectSlot(); // MISSING: can be removed with Qt5 using a functor in Object
    void setBoundingBox(bool); // MISSING: can be removed with Qt5 using a functor in Object
    void newRigidBodySlot(); // MISSING: can be removed with Qt5 using a functor in CompoundRigidBody
    void newObjectSlot(); // MISSING: can be removed with Qt5 using a functor in Group
    void saveFileSlot(); // MISSING: can be removed with Qt5 using a functor in Group
    void unloadFileSlot(); // MISSING: can be removed with Qt5 using a functor in Group
    void reloadFileSlot(); // MISSING: can be removed with Qt5 using a functor in Group
    void moveCameraWithSlot_RigidBody(); // MISSING: can be removed with Qt5 using a functor in RigidBody
    void moveCameraWithSlot_Nurbsdisk(); // MISSING: can be removed with Qt5 using a functor in NurbsDisk
};






class Editor : public QWidget, virtual public fmatvec::Atom {
  public:
    Editor(PropertyDialog *parent_, const QIcon &icon, const std::string &name);
  protected:
    PropertyDialog *dialog;
    void replaceObject();
    static void getSelAndCur(QTreeWidgetItem *item, std::queue<bool> &sel, std::queue<bool> &cur);
    static void setSelAndCur(QTreeWidgetItem *item, std::queue<bool> &sel, std::queue<bool> &cur);
    static void unsetClone(Object *obj);
};





/*! A Editor of type boolean */
class BoolEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor. */
    BoolEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name, const std::string &qtObjectName,
               bool replaceObjOnChange_=true);

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, bool (OMBVClass::*getter)(), void (OMBVClass::*setter)(bool));

    /* return this boolean Editor as an checkable action */
    QAction *getAction() { return action; }

  signals:
    void stateChanged(bool state);

  protected slots:
    void valueChangedSlot(int);
    void actionChangedSlot();

  protected:
    QCheckBox *checkbox;
    std::function<bool ()> ombvGetter;
    std::function<void (bool)> ombvSetter;
    QAction *action;
    bool replaceObjOnChange;
};





/*! A Editor of type double */
class FloatEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor. */
    FloatEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name);

    /*! Set the valid range of the double value */
    void setRange(double min, double max) { spinBox->blockSignals(true); spinBox->setRange(min, max); spinBox->blockSignals(false); }

    /*! Set step size of the double value */
    void setStep(double step) { spinBox->blockSignals(true); spinBox->setSingleStep(step); spinBox->blockSignals(false); }

//    /*! Set the special value at min */
//    void setNaNText(const std::string &value) { spinBox->blockSignals(true); spinBox->setSpecialValueText(value.c_str()); spinBox->blockSignals(false); }

    /*! Set the suffix to display */
    void setSuffix(const QString &value) { spinBox->blockSignals(true); spinBox->setSuffix(value); spinBox->blockSignals(false); }

    /*! Set the conversion factor between the display value and the real value */
    void setFactor(const double value) { factor=value; }

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, double (OMBVClass::*getter)(), void (OMBVClass::*setter)(double));

  protected slots:
    void valueChangedSlot(double);

  protected:
    double factor;
    QDoubleSpinBox *spinBox;
    std::function<double ()> ombvGetter;
    std::function<void (double)> ombvSetter;
};





/*! A Editor of type double for a matrix or vector */
class FloatMatrixEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor.
     * Use 0 for rows/cols to defined a arbitary size in this dimension.*/
    FloatMatrixEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name, unsigned int rows_, unsigned int cols_);

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface.
     * Vector version. */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<double> (OMBVClass::*getter)(),
                                               void (OMBVClass::*setter)(const std::vector<double>&));

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface.
     * Matrix version. */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<std::vector<double> > (OMBVClass::*getter)(),
                                               void (OMBVClass::*setter)(const std::vector<std::vector<double> >&));

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface.
     * std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > version. */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_,
      std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > (OMBVClass::*getter)(),
      void (OMBVClass::*setter)(const std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > &));

  protected slots:
    void addRowSlot(); // calls valueChanged (also see addRow)
    void removeRowSlot(); // calls valueChanged
    void addColumnSlot(); // calls valueChanged (also see addColumn)
    void removeColumnSlot(); // calls valueChanged
    void valueChangedSlot();

  protected:
    void addRow(); // does not call valueChanged (also see addRowSlot)
    void addColumn(); // does not call valueChanged (also see addColumnSlot)
    unsigned int rows, cols;
    QTableWidget *table;
    std::function<std::vector<double> ()> ombvGetterVector;
    std::function<void (const std::vector<double>&)> ombvSetterVector;
    std::function<std::vector<std::vector<double> > ()> ombvGetterMatrix;
    std::function<void (const std::vector<std::vector<double> >&)> ombvSetterMatrix;
    std::function<std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > ()> ombvGetterPolygonPoint;
    std::function<void (const std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > &)> ombvSetterPolygonPoint;
};





/*! A Editor of type int */
class IntEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor. */
    IntEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name);

    /*! Set the valid range of the double value */
    void setRange(int min, int max) { spinBox->blockSignals(true); spinBox->setRange(min, max); spinBox->blockSignals(false); }

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, int (OMBVClass::*getter)(), void (OMBVClass::*setter)(int));
 
    /*! unsigned int version of setOpenMBVParameter */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, unsigned int (OMBVClass::*getter)(), void (OMBVClass::*setter)(unsigned int));

  protected slots:
    void valueChangedSlot(int);

  protected:
    QSpinBox *spinBox;
    std::function<int ()> ombvGetter;
    std::function<void (int)> ombvSetter;
};





/*! A Editor of type string */
class StringEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor. */
    StringEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name);

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::string (OMBVClass::*getter)(), void (OMBVClass::*setter)(std::string));

  protected slots:
    void valueChangedSlot(const QString&);

  protected:
    QLineEdit *lineEdit;
    std::function<std::string ()> ombvGetter;
    std::function<void (std::string)> ombvSetter;
};





/*! A Editor of type enum */
class ComboBoxEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor. */
    ComboBoxEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name,
      const std::vector<std::tuple<int, std::string, QIcon, std::string> > &list);

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass, class OMBVEnum>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, OMBVEnum (OMBVClass::*getter)(), void (OMBVClass::*setter)(OMBVEnum));

    /* return this ComboBox Editor as an ActionGroup */
    QActionGroup *getActionGroup() { return actionGroup; }

  protected slots:
    void valueChangedSlot(int);
    void actionChangedSlot(QAction* action);

  protected:
    QComboBox *comboBox;
    std::function<int ()> ombvGetter;
    std::function<void (int)> ombvSetter;
    QActionGroup *actionGroup;
};





/*! A Editor of type 3xdouble */
class Vec3fEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor. */
    Vec3fEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name);

    /*! Set the valid range of the double values */
    void setRange(double min, double max) {
      for(int i=0; i<3; i++) {
        spinBox[i]->blockSignals(true);
        spinBox[i]->setRange(min, max);
        spinBox[i]->blockSignals(false);
      }
    }

    /*! Set step size of the double values */
    void setStep(double step) {
      for(int i=0; i<3; i++) {
        spinBox[i]->blockSignals(true);
        spinBox[i]->setSingleStep(step);
        spinBox[i]->blockSignals(false);
      }
    }

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<double> (OMBVClass::*getter)(),
                                               void (OMBVClass::*setter)(double x, double y, double z));

  protected slots:
    void valueChangedSlot();

  protected:
    QDoubleSpinBox *spinBox[3];
    std::function<std::vector<double> ()> ombvGetter;
    std::function<void (double, double, double)> ombvSetter;
};



/*! A Editor of type color */
class ColorEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor. */
    ColorEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name, bool showResetHueButton=false);

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<double> (OMBVClass::*getter)(),
                                               void (OMBVClass::*setter)(double h, double s, double v));

  protected slots:
    void valueChangedSlot();
    void showDialog();
    void resetHue();

  protected:
    QColorDialog *colorDialog;
    std::function<std::vector<double> ()> ombvGetter;
    std::function<void (double, double, double)> ombvSetter;
};





/*! A special Editor to edit a (mechanical) relative translation and rotation between two frames incling a interactive Dragger.
 * Note: This Editor does NOT delete and renew the Object at each value change. */
class TransRotEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor.
     * soTranslation_ and soRotation_ is syncronized with this Editor */
    TransRotEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name);
    ~TransRotEditor();

    void setGroupMembers(SoGroup *grp);

    /*! Set step size of the translation values (rotation is 10deg) */
    void setStep(double step) {
      for(int i=0; i<3; i++) {
        spinBox[i]->blockSignals(true);
        spinBox[i]->setSingleStep(step);
        spinBox[i]->blockSignals(false);
      }
    }

    /*! OpenMBVCppInterface syncronization.
     * Use *Getter and *Setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<double> (OMBVClass::*transGetter)(), 
                                               void (OMBVClass::*transSetter)(double x, double y, double z),
                                               std::vector<double> (OMBVClass::*rotGetter)(),
                                               void (OMBVClass::*rotSetter)(double x, double y, double z),
                                               bool (OMBVClass::*draggerGetter)(),
                                               void (OMBVClass::*draggerSetter)(bool b));

  protected slots:
    void valueChangedSlot();
    void draggerSlot(int state);

  protected:
    QDoubleSpinBox *spinBox[6];
    SoTranslation *soTranslation;
    SoRotation *soRotation;
    std::function<std::vector<double> ()> ombvTransGetter, ombvRotGetter;
    std::function<void (double, double, double)> ombvTransSetter, ombvRotSetter;
    std::function<bool ()> ombvDraggerGetter;
    std::function<void (bool)> ombvDraggerSetter;
    SoCenterballDragger *soDragger;
    SoSwitch *soDraggerSwitch;
    QCheckBox *draggerCheckBox;
    static void draggerMoveCB(void *, SoDragger *dragger_);
    static void draggerFinishedCB(void *, SoDragger *dragger_);
};





/*! A dummy Editor of a not available type */
class NotAvailableEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor. */
    NotAvailableEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name);
};






/******************************************************************/





template<class OMBVClass>
void BoolEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, bool (OMBVClass::*getter)(), void (OMBVClass::*setter)(bool)) {
  ombvGetter=std::bind(getter, ombv_);
  ombvSetter=std::bind(setter, ombv_, std::placeholders::_1);
  checkbox->blockSignals(true);
  checkbox->setCheckState(ombvGetter()?Qt::Checked:Qt::Unchecked);
  checkbox->blockSignals(false);
  action->blockSignals(true);
  action->setChecked(ombvGetter()?Qt::Checked:Qt::Unchecked);
  action->blockSignals(false);
}





template<class OMBVClass>
void FloatEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, double (OMBVClass::*getter)(), void (OMBVClass::*setter)(double)) {
  ombvGetter=std::bind(getter, ombv_);
  ombvSetter=std::bind(setter, ombv_, std::placeholders::_1);
  spinBox->blockSignals(true);
  if(spinBox->specialValueText()=="" || !std::isnan(ombvGetter()))
    spinBox->setValue(ombvGetter()/factor);
  else
    spinBox->setValue(spinBox->minimum());
  spinBox->blockSignals(false);
}





template<class OMBVClass>
void FloatMatrixEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<double> (OMBVClass::*getter)(),
                                                              void (OMBVClass::*setter)(const std::vector<double>&)) {
  // set functions
  ombvGetterVector=std::bind(getter, ombv_);
  ombvSetterVector=std::bind(setter, ombv_, std::placeholders::_1);
  // asserts
  assert((cols==1 && (rows==0 || rows==ombvGetterVector().size())) ||
         (rows==1 && (cols==0 || cols==ombvGetterVector().size())));
  // create cells
  if(cols==1) {
    table->setColumnCount(1);
    for(unsigned int r=0; r<ombvGetterVector().size(); r++)
      addRow();
  }
  else { // rows==1
    table->setRowCount(1);
    for(unsigned int c=0; c<ombvGetterVector().size(); c++)
      addColumn();
  }
  // set cell values
  for(unsigned int i=0; i<ombvGetterVector().size(); i++) {
    QDoubleSpinBox *spinBox;
    if(cols==1)
      spinBox=static_cast<QDoubleSpinBox*>(table->cellWidget(i, 0));
    else
      spinBox=static_cast<QDoubleSpinBox*>(table->cellWidget(0, i));
    spinBox->blockSignals(true);
    spinBox->setValue(ombvGetterVector()[i]);
    spinBox->blockSignals(false);
  }
}

template<class OMBVClass>
void FloatMatrixEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<std::vector<double> > (OMBVClass::*getter)(),
                                                              void (OMBVClass::*setter)(const std::vector<std::vector<double> >&)) {
  // set functions
  ombvGetterMatrix=std::bind(getter, ombv_);
  ombvSetterMatrix=std::bind(setter, ombv_, std::placeholders::_1);
  // asserts
  assert(rows==0 || rows==ombvGetterMatrix().size());
  assert(cols==0 || cols==ombvGetterMatrix()[0].size());
  // create cells
  table->setRowCount(ombvGetterMatrix().size());
  for(unsigned int c=0; c<ombvGetterMatrix()[0].size(); c++)
    addColumn();
  // set cell values
  for(unsigned int r=0; r<ombvGetterMatrix().size(); r++)
    for(unsigned int c=0; c<ombvGetterMatrix()[r].size(); c++) {
      QDoubleSpinBox *spinBox;
      spinBox=static_cast<QDoubleSpinBox*>(table->cellWidget(r, c));
      spinBox->blockSignals(true);
      spinBox->setValue(ombvGetterMatrix()[r][c]);
      spinBox->blockSignals(false);
    }
}


template<class OMBVClass>
void FloatMatrixEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_,
  std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > (OMBVClass::*getter)(),
  void (OMBVClass::*setter)(const std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > &)) {
  // set functions
  ombvGetterPolygonPoint=std::bind(getter, ombv_);
  ombvSetterPolygonPoint=std::bind(setter, ombv_, std::placeholders::_1);
  // asserts
  assert(cols==3);
  assert(rows==0 || rows==ombvGetterMatrix().size());
  // create cells
  table->setColumnCount(3);
  for(unsigned int r=0; r<(ombvGetterPolygonPoint()?ombvGetterPolygonPoint()->size():0); r++)
    addRow();
  // set cell values
  for(unsigned int r=0; r<(ombvGetterPolygonPoint()?ombvGetterPolygonPoint()->size():0); r++) {
    QDoubleSpinBox *spinBox;

    spinBox=static_cast<QDoubleSpinBox*>(table->cellWidget(r, 0));
    spinBox->blockSignals(true);
    spinBox->setValue((*ombvGetterPolygonPoint())[r]->getXComponent());
    spinBox->blockSignals(false);

    spinBox=static_cast<QDoubleSpinBox*>(table->cellWidget(r, 1));
    spinBox->blockSignals(true);
    spinBox->setValue((*ombvGetterPolygonPoint())[r]->getYComponent());
    spinBox->blockSignals(false);

    spinBox=static_cast<QDoubleSpinBox*>(table->cellWidget(r, 2));
    spinBox->blockSignals(true);
    spinBox->setValue((*ombvGetterPolygonPoint())[r]->getBorderValue());
    spinBox->setSingleStep(1);
    spinBox->setRange(0, 1);
    spinBox->setDecimals(0);
    spinBox->blockSignals(false);
  }
}




template<class OMBVClass>
void IntEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, int (OMBVClass::*getter)(), void (OMBVClass::*setter)(int)) {
  ombvGetter=std::bind(getter, ombv_);
  ombvSetter=std::bind(setter, ombv_, std::placeholders::_1);
  spinBox->blockSignals(true);
  spinBox->setValue(ombvGetter());
  spinBox->blockSignals(false);
}

template<class OMBVClass>
void IntEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, unsigned int (OMBVClass::*getter)(), void (OMBVClass::*setter)(unsigned int)) {
  setOpenMBVParameter(ombv_, reinterpret_cast<int (OMBVClass::*)()>(getter), reinterpret_cast<void (OMBVClass::*)(int)>(setter));
}





template<class OMBVClass>
void StringEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::string (OMBVClass::*getter)(), void (OMBVClass::*setter)(std::string)) {
  ombvGetter=std::bind(getter, ombv_);
  ombvSetter=std::bind(setter, ombv_, std::placeholders::_1);
  lineEdit->blockSignals(true);
  lineEdit->setText(ombvGetter().c_str());
  lineEdit->blockSignals(false);
}





template<class OMBVClass, class OMBVEnum>
void ComboBoxEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, OMBVEnum (OMBVClass::*getter)(), void (OMBVClass::*setter)(OMBVEnum)) {
  ombvGetter=std::bind(getter, ombv_);
  ombvSetter=std::bind(reinterpret_cast<void (OMBVClass::*)(int)>(setter), ombv_, std::placeholders::_1); // for the setter we have to cast the first argument from OMBVEnum to int
  comboBox->blockSignals(true);
  comboBox->setCurrentIndex(comboBox->findData(ombvGetter()));
  comboBox->blockSignals(false);
  QAction *action=actionGroup->actions()[comboBox->findData(ombvGetter())+1];
  action->blockSignals(true);
  action->setChecked(true);
  action->blockSignals(false);
}





template<class OMBVClass>
void Vec3fEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<double> (OMBVClass::*getter)(), void (OMBVClass::*setter)(double x, double y, double z)) {
  ombvGetter=std::bind(getter, ombv_);
  ombvSetter=std::bind(setter, ombv_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  std::vector<double> vec=ombvGetter();
  for(int i=0; i<3; i++) {
    spinBox[i]->blockSignals(true);
    spinBox[i]->setValue(vec[i]);
    spinBox[i]->blockSignals(false);
  }
}



template<class OMBVClass>
void ColorEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<double> (OMBVClass::*getter)(), void (OMBVClass::*setter)(double h, double s, double v)) {
  ombvGetter=std::bind(getter, ombv_);
  ombvSetter=std::bind(setter, ombv_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  std::vector<double> vec=ombvGetter();
  colorDialog->blockSignals(true);
  QColor color;
  color.setHsvF(vec[0], vec[1], vec[2]);
  colorDialog->setCurrentColor(color);
  colorDialog->blockSignals(false);
}





template<class OMBVClass>
void TransRotEditor::setOpenMBVParameter(std::shared_ptr<OMBVClass> &ombv_, std::vector<double> (OMBVClass::*transGetter)(),
                                                           void (OMBVClass::*transSetter)(double x, double y, double z),
                                                           std::vector<double> (OMBVClass::*rotGetter)(),
                                                           void (OMBVClass::*rotSetter)(double x, double y, double z),
                                                           bool (OMBVClass::*draggerGetter)(),
                                                           void (OMBVClass::*draggerSetter)(bool b)) {
  ombvTransGetter=std::bind(transGetter, ombv_);
  ombvTransSetter=std::bind(transSetter, ombv_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  ombvRotGetter=std::bind(rotGetter, ombv_);
  ombvRotSetter=std::bind(rotSetter, ombv_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  ombvDraggerGetter=std::bind(draggerGetter, ombv_);
  ombvDraggerSetter=std::bind(draggerSetter, ombv_, std::placeholders::_1);
  std::vector<double> trans=ombvTransGetter();
  std::vector<double> rot=ombvRotGetter();
  soTranslation->translation.setValue(trans[0], trans[1], trans[2]);
  soRotation->rotation=Utils::cardan2Rotation(SbVec3f(rot[0],rot[1],rot[2])).invert();
  soDraggerSwitch->whichChild.setValue(ombvDraggerGetter()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  for(int i=0; i<3; i++) {
    spinBox[i  ]->setValue(trans[i]);
    spinBox[i+3]->setValue(rot[i]*180/M_PI);
  }
  draggerCheckBox->setChecked(ombvDraggerGetter());
}

}

#endif
