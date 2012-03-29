#ifndef _EDITORS_H_
#define _EDITORS_H_

#include <QAction>
#include <Inventor/fields/SoSFFloat.h>
#include <Inventor/fields/SoMFFloat.h>
#include <Inventor/fields/SoSFInt32.h>
#include <Inventor/fields/SoSFVec3f.h>
#include <Inventor/fields/SoSFRotation.h>
#include <Inventor/nodes/SoSwitch.h>
#include <QDoubleSpinBox>
#include <QDockWidget>
#include <openmbvcppinterface/simpleparameter.h>
#include <boost/function.hpp>
#include <boost/bind.hpp>
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
    std::string groupName();
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
    template<class OMBV>
    void setOpenMBVParameter(OMBV *ombv_, bool (OMBV::*getter)(), void (OMBV::*setter)(bool));

    /*! Set the scene object which should be synced */
    void setSo(SoSFInt32 *soBool_) { soBool=soBool_; }

  protected slots:
    void valueChangedSlot(bool);
  protected:

    SoSFInt32 *soBool;
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

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBV>
    void setOpenMBVParameter(OMBV *ombv_, double (OMBV::*getter)(), void (OMBV::*setter)(OpenMBV::ScalarParameter));

    /*! Set the scene object which should be synced */
    void setSo(SoSFFloat *soValue_) { soValue=soValue_; }
    /*! Set the scene object which should be synced */
    void setSo(SoMFFloat *soValue_) { soValue=soValue_; }

  protected slots:
    void valueChangedSlot(double newValue);

  protected:
    void constructor(const std::string &name);
    QDoubleSpinBox *spinBox;
    SoField *soValue;
    boost::function<double ()> ombvGetter;
    boost::function<void (double)> ombvSetter;
    void setValue(double v) {
      if(soValue==NULL) return;
      if(soValue->isOfType(SoSFFloat::getClassTypeId()))
        static_cast<SoSFFloat*>(soValue)->setValue(v);
      else
        static_cast<SoMFFloat*>(soValue)->setValue(v);
    }
    double getValue() {
      if(soValue==NULL) return 0;
      if(soValue->isOfType(SoSFFloat::getClassTypeId()))
        return static_cast<SoSFFloat*>(soValue)->getValue();
      else
        return *(static_cast<SoMFFloat*>(soValue)->getValues(0));
    }
};





/*! A Editor of type 3xdouble */
class Vec3fEditor : public WidgetEditor {
  Q_OBJECT

  public:
    /*! Constructor. */
    Vec3fEditor(QObject *parent_, const QIcon &icon, const std::string &name);

    /*! Set the valid range of the double values */
    void setRange(double min, double max) { for(int i=0; i<3; i++) spinBox[i]->setRange(min, max); }

    /*! Set step size of the double values */
    void setStep(double step) { for(int i=0; i<3; i++) spinBox[i]->setSingleStep(step); }

    /*! OpenMBVCppInterface syncronization.
     * Use getter and setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBV>
    void setOpenMBVParameter(OMBV *ombv_, std::vector<double> (OMBV::*getter)(),
                                          void (OMBV::*setter)(double x, double y, double z));

    /*! Set the scene object which should be synced */
    void setSo(SoSFVec3f *soValue_) { soValue=soValue_; }
    /*! Set the scene object which should be synced */
    void setSo(SoSFFloat *soX_, SoSFFloat *soY_, SoSFFloat *soZ_) { so[0]=soX_; so[1]=soY_; so[2]=soZ_; }

  protected slots:
    void valueChangedSlot();

  protected:
    void constructor(const std::string &name);
    QDoubleSpinBox *spinBox[3];
    SoSFVec3f *soValue;
    SoSFFloat *so[3];
    boost::function<std::vector<double> ()> ombvGetter;
    boost::function<void (double, double, double)> ombvSetter;
};





/*! A special Editor to edit a (mechanical) relative translation and rotation between two frames incling a interactive Dragger. */
class TransRotEditor : public WidgetEditor {
  Q_OBJECT

  public:
    /*! Constructor.
     * soTranslation_ and soRotation_ is syncronized with this Editor */
    TransRotEditor(QObject *parent_, const QIcon &icon, const std::string &name, SoSFVec3f *soTranslation_, SoSFRotation *soRotation_);

    /*! Set step size of the translation values (rotation is 10deg) */
    void setStep(double step) { for(int i=0; i<3; i++) spinBox[i]->setSingleStep(step); }

    /*! OpenMBVCppInterface syncronization.
     * Use *Getter and *Setter of ombv_ to sync this Editor with OpenMBVCppInterface */
    template<class OMBV>
    void setOpenMBVParameter(OMBV *ombv_, std::vector<double> (OMBV::*transGetter)(), 
                                          void (OMBV::*transSetter)(double x, double y, double z),
                                          std::vector<double> (OMBV::*rotGetter)(),
                                          void (OMBV::*rotSetter)(double x, double y, double z));

    /*! Set the group this translation/rotation resizes in.
     * The Dragger must be added before the translation and rotation nodes referenced by the constructor. */
    void setDragger(SoGroup *draggerParent);

    /*! return a action to active the Dragger */
    QAction *getDraggerAction() { return draggerAction; }

  protected slots:
    void valueChangedSlot();
    void draggerSlot(bool newValue);

  protected:
    QDoubleSpinBox *spinBox[6];
    SoSFVec3f *soTranslation;
    SoSFRotation *soRotation;
    boost::function<std::vector<double> ()> ombvTransGetter, ombvRotGetter;
    boost::function<void (double, double, double)> ombvTransSetter, ombvRotSetter;
    SoCenterballDragger *soDragger;
    SoSwitch *soDraggerSwitch;
    QAction *draggerAction;
    static void draggerMoveCB(void *, SoDragger *dragger_);
    static void draggerFinishedCB(void *, SoDragger *dragger_);
};






/******************************************************************/





template<class OMBV>
void BoolEditor::setOpenMBVParameter(OMBV *ombv_, bool (OMBV::*getter)(), void (OMBV::*setter)(bool)) {
  ombvGetter=boost::bind(getter, ombv_);
  ombvSetter=boost::bind(setter, ombv_, _1);
  if(ombvGetter) {
    if(soBool) soBool->setValue(ombvGetter()?SO_SWITCH_ALL:SO_SWITCH_NONE);
    action->setChecked(ombvGetter());
  }
}





template<class OMBV>
void FloatEditor::setOpenMBVParameter(OMBV *ombv_, double (OMBV::*getter)(), void (OMBV::*setter)(OpenMBV::ScalarParameter)) {
  ombvGetter=boost::bind(getter, ombv_);
  ombvSetter=boost::bind(setter, ombv_, _1);
  if(ombvGetter) {
    setValue(ombvGetter());
    spinBox->blockSignals(true);
    spinBox->setValue(ombvGetter());
    spinBox->blockSignals(false);
  }
}





template<class OMBV>
void Vec3fEditor::setOpenMBVParameter(OMBV *ombv_, std::vector<double> (OMBV::*getter)(), void (OMBV::*setter)(double x, double y, double z)) {
  ombvGetter=boost::bind(getter, ombv_);
  ombvSetter=boost::bind(setter, ombv_, _1, _2, _3);
  if(ombvGetter) {
    std::vector<double> vec=ombvGetter();
    if(soValue)
      soValue->setValue(vec[0], vec[1], vec[2]);
    else if(so[0]) {
      so[0]->setValue(vec[0]);
      so[1]->setValue(vec[1]);
      so[2]->setValue(vec[2]);
    }
    for(int i=0; i<3; i++)
      spinBox[i]->setValue(vec[i]);
  }
}





template<class OMBV>
void TransRotEditor::setOpenMBVParameter(OMBV *ombv_, std::vector<double> (OMBV::*transGetter)(),
                                                      void (OMBV::*transSetter)(double x, double y, double z),
                                                      std::vector<double> (OMBV::*rotGetter)(),
                                                      void (OMBV::*rotSetter)(double x, double y, double z)) {
  ombvTransGetter=boost::bind(transGetter, ombv_);
  ombvTransSetter=boost::bind(transSetter, ombv_, _1, _2, _3);
  ombvRotGetter=boost::bind(rotGetter, ombv_);
  ombvRotSetter=boost::bind(rotSetter, ombv_, _1, _2, _3);
  if(ombvTransGetter && ombvRotGetter) {
    std::vector<double> trans=ombvTransGetter();
    std::vector<double> rot=ombvRotGetter();
    soTranslation->setValue(trans[0], trans[1], trans[2]);
    *soRotation=Utils::cardan2Rotation(SbVec3f(rot[0],rot[1],rot[2])).invert();
    for(int i=0; i<3; i++) {
      spinBox[i  ]->setValue(trans[i]);
      spinBox[i+3]->setValue(rot[i]*180/M_PI);
    }
  }
}

#endif
