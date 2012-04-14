#ifndef _EDITORS_H_
#define _EDITORS_H_

#include <QAction>
#include <Inventor/fields/SoSFFloat.h>
#include <Inventor/fields/SoMFFloat.h>
#include <Inventor/fields/SoSFInt32.h>
#include <Inventor/fields/SoSFVec3f.h>
#include <Inventor/fields/SoSFRotation.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoTranslation.h>
#include <QDoubleSpinBox>
#include <QDockWidget>
#include <QComboBox>
#include <QLineEdit>
#include <openmbvcppinterface/simpleparameter.h>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/assign/list_of.hpp>
#include <utils.h>

class SoCenterballDragger;
class SoDragger;
class QGridLayout;
class QVBoxLayout;
class QScrollArea;

/*! Base class for all Editors */
class Editor : public QObject {
  Q_OBJECT

  public:
    /*! Constructor.
     * parent_ should be a of type Object if possible.
     * icon is displayed by the action.
     * name_ is displayed by the action (a '...' is added for a WidgetEditor) */
    Editor(QObject *parent_, const QIcon &icon, const std::string &name_);

    /*! Get the action to activate this Editor */
    QAction *getAction() { return action; }

  protected:
    std::string name;
    QObject *parent;
    QAction *action;
    std::string sortName();
    void replaceObject();
};





/*! A Editor of type boolean */
class BoolEditor : public Editor {
  Q_OBJECT

  public:
    /*! Constructor. */
    BoolEditor(QObject *parent_, const QIcon &icon, const std::string &name);

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(OMBVClass *ombv_, bool (OMBVClass::*getter)(), void (OMBVClass::*setter)(bool));

  protected slots:
    void valueChangedSlot(bool);

  protected:
    boost::function<bool ()> ombvGetter;
    boost::function<void (bool)> ombvSetter;
};





/*! Base class for all WidgetEditors: a Editor which required a more complex layout than a simple action (like bools) */
class WidgetEditor : public Editor {
  Q_OBJECT
  friend class WidgetEditorCollector;

  public:
    WidgetEditor(QObject *parent_, const QIcon &icon, const std::string &name);
    ~WidgetEditor();

  protected slots:
    void actionClickedSlot(bool newValue);

  protected:
    QWidget *widget;
    QGridLayout *layout;
};





/*! A QDockWidget which collects and displays all WidgetEditors being currently activated */
class WidgetEditorCollector : public QDockWidget {
  friend class WidgetEditor;
  protected:
    WidgetEditorCollector();
    static WidgetEditorCollector *instance;
    std::multimap<std::string, WidgetEditor*> handledEditors;
    void updateLayout();
    QVBoxLayout *layout;
    QScrollArea *scrollArea;

  public:
    /* Get a singleton instance of WidgetEditorCollector */
    static WidgetEditorCollector *getInstance();

    /* Add a WidgetEditor */
    void addEditor(WidgetEditor *editor);

    /* Remove a WidgetEditor */
    void removeEditor(WidgetEditor *editor);
};





/*! A Editor of type double */
class FloatEditor : public WidgetEditor {
  Q_OBJECT

  public:
    /*! Constructor. */
    FloatEditor(QObject *parent_, const QIcon &icon, const std::string &name);

    /*! Set the valid range of the double value */
    void setRange(double min, double max) { spinBox->blockSignals(true); spinBox->setRange(min, max); spinBox->blockSignals(false); }

    /*! Set step size of the double value */
    void setStep(double step) { spinBox->blockSignals(true); spinBox->setSingleStep(step); spinBox->blockSignals(false); }

    /*! Set the special value at min */
    void setNaNText(const std::string &value) { spinBox->blockSignals(true); spinBox->setSpecialValueText(value.c_str()); spinBox->blockSignals(false); }

    /*! Set the suffix to display */
    void setSuffix(const QString &value) { spinBox->blockSignals(true); spinBox->setSuffix(value); spinBox->blockSignals(false); }

    /*! Set the conversion factor between the display value and the real value */
    void setFactor(const double value) { factor=value; }

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(OMBVClass *ombv_, double (OMBVClass::*getter)(), void (OMBVClass::*setter)(OpenMBV::ScalarParameter));

  protected slots:
    void valueChangedSlot(double);

  protected:
    double factor;
    QDoubleSpinBox *spinBox;
    boost::function<double ()> ombvGetter;
    boost::function<void (double)> ombvSetter;
};





/*! A Editor of type int */
class IntEditor : public WidgetEditor {
  Q_OBJECT

  public:
    /*! Constructor. */
    IntEditor(QObject *parent_, const QIcon &icon, const std::string &name);

    /*! Set the valid range of the double value */
    void setRange(int min, int max) { spinBox->blockSignals(true); spinBox->setRange(min, max); spinBox->blockSignals(false); }

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(OMBVClass *ombv_, int (OMBVClass::*getter)(), void (OMBVClass::*setter)(int));
 
    /*! unsigned int version of setOpenMBVParameter */
    template<class OMBVClass>
    void setOpenMBVParameter(OMBVClass *ombv_, unsigned int (OMBVClass::*getter)(), void (OMBVClass::*setter)(unsigned int));

  protected slots:
    void valueChangedSlot(int);

  protected:
    QSpinBox *spinBox;
    boost::function<int ()> ombvGetter;
    boost::function<void (int)> ombvSetter;
};





/*! A Editor of type string */
class StringEditor : public WidgetEditor {
  Q_OBJECT

  public:
    /*! Constructor. */
    StringEditor(QObject *parent_, const QIcon &icon, const std::string &name);

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass>
    void setOpenMBVParameter(OMBVClass *ombv_, std::string (OMBVClass::*getter)(), void (OMBVClass::*setter)(std::string));

  protected slots:
    void valueChangedSlot();

  protected:
    QLineEdit *lineEdit;
    boost::function<std::string ()> ombvGetter;
    boost::function<void (std::string)> ombvSetter;
};





/*! A Editor of type enum */
class ComboBoxEditor : public WidgetEditor {
  Q_OBJECT

  public:
    /*! Constructor. */
    ComboBoxEditor(QObject *parent_, const QIcon &icon, const std::string &name,
      const std::vector<boost::tuple<int, std::string, QIcon> > &list);

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBVClass, class OMBVEnum>
    void setOpenMBVParameter(OMBVClass *ombv_, OMBVEnum (OMBVClass::*getter)(), void (OMBVClass::*setter)(OMBVEnum));

  protected slots:
    void valueChangedSlot(int);

  protected:
    QComboBox *comboBox;
    boost::function<int ()> ombvGetter;
    boost::function<void (int)> ombvSetter;
};





/*! A Editor of type 3xdouble */
class Vec3fEditor : public WidgetEditor {
  Q_OBJECT

  public:
    /*! Constructor. */
    Vec3fEditor(QObject *parent_, const QIcon &icon, const std::string &name);

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
    void setOpenMBVParameter(OMBVClass *ombv_, std::vector<double> (OMBVClass::*getter)(),
                                               void (OMBVClass::*setter)(double x, double y, double z));

  protected slots:
    void valueChangedSlot();

  protected:
    QDoubleSpinBox *spinBox[3];
    boost::function<std::vector<double> ()> ombvGetter;
    boost::function<void (double, double, double)> ombvSetter;
};





/*! A special Editor to edit a (mechanical) relative translation and rotation between two frames incling a interactive Dragger.
 * Note: This Editor does NOT delete and renew the Object at each value change. */
class TransRotEditor : public WidgetEditor {
  Q_OBJECT

  public:
    /*! Constructor.
     * soTranslation_ and soRotation_ is syncronized with this Editor */
    TransRotEditor(QObject *parent_, const QIcon &icon, const std::string &name, SoGroup *grp);

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
    void setOpenMBVParameter(OMBVClass *ombv_, std::vector<double> (OMBVClass::*transGetter)(), 
                                               void (OMBVClass::*transSetter)(double x, double y, double z),
                                               std::vector<double> (OMBVClass::*rotGetter)(),
                                               void (OMBVClass::*rotSetter)(double x, double y, double z));

    /*! return a action to active the Dragger */
    QAction *getDraggerAction() { return draggerAction; }

  protected slots:
    void valueChangedSlot();
    void draggerSlot(bool newValue);

  protected:
    QDoubleSpinBox *spinBox[6];
    SoTranslation *soTranslation;
    SoRotation *soRotation;
    boost::function<std::vector<double> ()> ombvTransGetter, ombvRotGetter;
    boost::function<void (double, double, double)> ombvTransSetter, ombvRotSetter;
    SoCenterballDragger *soDragger;
    SoSwitch *soDraggerSwitch;
    QAction *draggerAction;
    static void draggerMoveCB(void *, SoDragger *dragger_);
    static void draggerFinishedCB(void *, SoDragger *dragger_);
};






/******************************************************************/





template<class OMBVClass>
void BoolEditor::setOpenMBVParameter(OMBVClass *ombv_, bool (OMBVClass::*getter)(), void (OMBVClass::*setter)(bool)) {
  ombvGetter=boost::bind(getter, ombv_);
  ombvSetter=boost::bind(setter, ombv_, _1);
  if(ombvGetter) {
    action->blockSignals(true);
    action->setChecked(ombvGetter());
    action->blockSignals(false);
  }
}





template<class OMBVClass>
void FloatEditor::setOpenMBVParameter(OMBVClass *ombv_, double (OMBVClass::*getter)(), void (OMBVClass::*setter)(OpenMBV::ScalarParameter)) {
  ombvGetter=boost::bind(getter, ombv_);
  ombvSetter=boost::bind(setter, ombv_, _1);
  if(ombvGetter) {
    spinBox->blockSignals(true);
    if(spinBox->specialValueText()=="" || !isnan(ombvGetter()))
      spinBox->setValue(ombvGetter()/factor);
    else
      spinBox->setValue(spinBox->minimum());
    spinBox->blockSignals(false);
  }
}





template<class OMBVClass>
void IntEditor::setOpenMBVParameter(OMBVClass *ombv_, int (OMBVClass::*getter)(), void (OMBVClass::*setter)(int)) {
  ombvGetter=boost::bind(getter, ombv_);
  ombvSetter=boost::bind(setter, ombv_, _1);
  if(ombvGetter) {
    spinBox->blockSignals(true);
    spinBox->setValue(ombvGetter());
    spinBox->blockSignals(false);
  }
}

template<class OMBVClass>
void IntEditor::setOpenMBVParameter(OMBVClass *ombv_, unsigned int (OMBVClass::*getter)(), void (OMBVClass::*setter)(unsigned int)) {
  setOpenMBVParameter(ombv_, reinterpret_cast<int (OMBVClass::*)()>(getter), reinterpret_cast<void (OMBVClass::*)(int)>(setter));
}





template<class OMBVClass>
void StringEditor::setOpenMBVParameter(OMBVClass *ombv_, std::string (OMBVClass::*getter)(), void (OMBVClass::*setter)(std::string)) {
  ombvGetter=boost::bind(getter, ombv_);
  ombvSetter=boost::bind(setter, ombv_, _1);
  if(ombvGetter) {
    lineEdit->blockSignals(true);
    lineEdit->setText(ombvGetter().c_str());
    lineEdit->blockSignals(false);
  }
}





template<class OMBVClass, class OMBVEnum>
void ComboBoxEditor::setOpenMBVParameter(OMBVClass *ombv_, OMBVEnum (OMBVClass::*getter)(), void (OMBVClass::*setter)(OMBVEnum)) {
  ombvGetter=boost::bind(getter, ombv_);
  ombvSetter=boost::bind(reinterpret_cast<void (OMBVClass::*)(int)>(setter), ombv_, _1); // for the setter we have to cast the first argument from OMBVEnum to int
  if(ombvGetter) {
    comboBox->blockSignals(true);
    comboBox->setCurrentIndex(comboBox->findData(ombvGetter()));
    comboBox->blockSignals(false);
  }
}





template<class OMBVClass>
void Vec3fEditor::setOpenMBVParameter(OMBVClass *ombv_, std::vector<double> (OMBVClass::*getter)(), void (OMBVClass::*setter)(double x, double y, double z)) {
  ombvGetter=boost::bind(getter, ombv_);
  ombvSetter=boost::bind(setter, ombv_, _1, _2, _3);
  if(ombvGetter) {
    std::vector<double> vec=ombvGetter();
    for(int i=0; i<3; i++) {
      spinBox[i]->blockSignals(true);
      spinBox[i]->setValue(vec[i]);
      spinBox[i]->blockSignals(false);
    }
  }
}





template<class OMBVClass>
void TransRotEditor::setOpenMBVParameter(OMBVClass *ombv_, std::vector<double> (OMBVClass::*transGetter)(),
                                                           void (OMBVClass::*transSetter)(double x, double y, double z),
                                                           std::vector<double> (OMBVClass::*rotGetter)(),
                                                           void (OMBVClass::*rotSetter)(double x, double y, double z)) {
  ombvTransGetter=boost::bind(transGetter, ombv_);
  ombvTransSetter=boost::bind(transSetter, ombv_, _1, _2, _3);
  ombvRotGetter=boost::bind(rotGetter, ombv_);
  ombvRotSetter=boost::bind(rotSetter, ombv_, _1, _2, _3);
  if(ombvTransGetter && ombvRotGetter) {
    std::vector<double> trans=ombvTransGetter();
    std::vector<double> rot=ombvRotGetter();
    soTranslation->translation.setValue(trans[0], trans[1], trans[2]);
    soRotation->rotation=Utils::cardan2Rotation(SbVec3f(rot[0],rot[1],rot[2])).invert();
    for(int i=0; i<3; i++) {
      spinBox[i  ]->setValue(trans[i]);
      spinBox[i+3]->setValue(rot[i]*180/M_PI);
    }
  }
}

#endif
