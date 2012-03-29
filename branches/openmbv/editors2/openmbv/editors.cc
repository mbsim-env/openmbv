#include <editors.h>
#include <Inventor/draggers/SoCenterballDragger.h>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QScrollArea>
#include <QPushButton>
#include <QApplication>
#include <cfloat>
#include <mainwindow.h>
#include <Inventor/nodes/SoSurroundScale.h>

using namespace std;





Editor::Editor(QObject *parent_, const QIcon& icon, const string &name_) : QObject(parent_), name(name_), parent(parent_) {
  // create the action (with a default icon if none is given)
  if(icon.isNull())
    action=new QAction(Utils::QIconCached(":/editor.svg"), "", this);
  else
    action=new QAction(icon, "", this);
  action->setObjectName((string(parent->metaObject()->className())+"::"+name).c_str());
}





BoolEditor::BoolEditor(QObject *parent_, const QIcon& icon, const string &name, SoSFInt32 *soBool_) : Editor(parent_, icon, name), soBool(soBool_) {
  // make the action from Editor checkable and use this as bool editor
  connect(action, SIGNAL(toggled(bool)), this, SLOT(valueChangedSlot(bool)));
  action->setText(name.c_str());
  action->setCheckable(true);
}

void BoolEditor::valueChangedSlot(bool checked) {
  soBool->setValue(checked?SO_SWITCH_ALL:SO_SWITCH_NONE); // set so
  if(ombvSetter) ombvSetter(checked); // set OpenMBV
  emit valueChanged(); // trigger valueChanged signal of Editor
}





WidgetEditor::WidgetEditor(QObject *parent_, const QIcon& icon, const string &name) : Editor(parent_, icon, name) {
  // make the action checkable to indicate if Widget of this WidgetEditor is active
  action->setText((name+"...").c_str());
  action->setCheckable(true);
  connect(action, SIGNAL(toggled(bool)), this, SLOT(actionClickedSlot(bool)));
  // create Widget with GridLayout to hold all elements the editor requires
  widget=new QWidget;
  widget->setContentsMargins(0, -10, -10, -10);
  layout=new QGridLayout;
  widget->setLayout(layout);
  // add a close button to uncheck the action (close the Widget of this editor)
  QPushButton *close=new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DockWidgetCloseButton), "");
  layout->addWidget(close, 0, 0);
  close->setFixedSize(close->sizeHint());
  connect(close, SIGNAL(clicked()), action, SLOT(toggle()));
}

WidgetEditor::~WidgetEditor() {
  // remove the widget from the WidgetEditorCollector and deallocate the widgete including all childs
  WidgetEditorCollector::getInstance()->removeEditor(this);
  delete widget;
}

void WidgetEditor::actionClickedSlot(bool newValue) {
  // add or remove the widget depending of the "check" flag of the corrosponding action
  if(newValue)
    WidgetEditorCollector::getInstance()->addEditor(this);
  else
    WidgetEditorCollector::getInstance()->removeEditor(this);
}





WidgetEditorCollector *WidgetEditorCollector::instance=NULL;

WidgetEditorCollector::WidgetEditorCollector() : QDockWidget() {
  // a DockWidget with a scroll aero to hold all WidgetEditors by a QVBoxLayout
  setWindowTitle("Property Editor");
  setObjectName(windowTitle());
  scrollArea=new QScrollArea;
  scrollArea->setWidgetResizable(true);
  setWidget(scrollArea);
  QWidget *scrollWidget=new QWidget;
  scrollArea->setWidget(scrollWidget);
  layout=new QVBoxLayout;
  scrollWidget->setLayout(layout);
  // add the Dock to the MainWindow
  MainWindow::getInstance()->addDockWidget(Qt::BottomDockWidgetArea,this);
}

WidgetEditorCollector *WidgetEditorCollector::getInstance() {
  // create a singelton instance
  if(instance==NULL)
    instance=new WidgetEditorCollector;
  return instance;
}

void WidgetEditorCollector::addEditor(WidgetEditor *editor) {
  // sort all WidgetEditors by Path (if parent is a Object) and then by the class name
  string sort;
  Object *obj=dynamic_cast<Object*>(editor->parent);
  if(obj)
    sort=obj->getPath()+": ";
  sort+=editor->parent->metaObject()->className();
  handledEditors.insert(pair<string, WidgetEditor*>(sort, editor)); // stores all active WidgetEditors
  // update Dock using all active WidgetEditors given in handledEditors 
  updateLayout();
  // open the editor and show newly added widget
  scrollArea->ensureWidgetVisible(editor->widget, 0, 0);
  show();
}

void WidgetEditorCollector::removeEditor(WidgetEditor *editor) {
  // delete editor from the list of active WidgetEditor
  for(multimap<string, WidgetEditor*>::iterator i=handledEditors.begin(); i!=handledEditors.end(); i++)
    if(i->second==editor) {
      handledEditors.erase(i);
      break;
    }
  // update Dock using all active WidgetEditors given in handledEditors 
  updateLayout();
  // close Dock if all WidgetEditors are removed
  if(layout->count()<=0)
    close();
}

void WidgetEditorCollector::updateLayout() {
  // remove all items from the layout
  for(int i=layout->count()-1; i>=0; i--) { // loop over the QGroupBox'es
    QWidget *item=layout->itemAt(i)->widget();
    for(int j=item->layout()->count()-1; j>=0; j--) { // loop over the Widget's in the QGroupBox
      QWidget *innerItem=item->layout()->itemAt(j)->widget();
      item->layout()->removeWidget(innerItem); // remove innerItem
      innerItem->setParent(NULL); // and set parent to NULL to prevent a delete by the parent
    }
    layout->removeWidget(item); // remove item
    delete item; // and delete
  }
  // add all items in handledEditors
  string oldSort="";
  QGroupBox *group=NULL;
  for(multimap<string, WidgetEditor*>::iterator i=handledEditors.begin(); i!=handledEditors.end(); i++) {
    if(oldSort!=i->first) { // add a new QGroupBox for each new Path
      group=new QGroupBox(i->first.c_str());
      layout->addWidget(group);
      group->setLayout(new QVBoxLayout);
    }
    oldSort=i->first;
    group->layout()->addWidget(i->second->widget);
  }
}





FloatEditor::FloatEditor(QObject *parent_, const QIcon& icon, const string &name, SoSFFloat *soValue_) : WidgetEditor(parent_, icon, name), soValue(soValue_) {
  constructor(name);
}

FloatEditor::FloatEditor(QObject *parent_, const QIcon& icon, const string &name, SoMFFloat *soValue_) : WidgetEditor(parent_, icon, name), soValue(soValue_) {
  constructor(name);
}

void FloatEditor::constructor(const string &name) {
  // add the label and a spinbox for the value
  spinBox=new QDoubleSpinBox;
  spinBox->setSingleStep(0.01);
  spinBox->setRange(-DBL_MAX, DBL_MAX);
  spinBox->setDecimals(6);
  spinBox->setFixedWidth(QFontMetrics(spinBox->font()).width("-888.888888000"));
  connect(spinBox, SIGNAL(valueChanged(double)), this, SLOT(valueChangedSlot(double)));
  layout->addWidget(new QLabel((name+":").c_str()), 0, 1);
  layout->addWidget(spinBox, 0, 2);
}

void FloatEditor::valueChangedSlot(double newValue) {
  setValue(newValue); // set so
  if(ombvSetter) ombvSetter(newValue); // set OpenMBV
  emit valueChanged(); // trigger valueChanged signal of Editor
}





Vec3fEditor::Vec3fEditor(QObject *parent_, const QIcon& icon, const string &name, SoSFVec3f *soValue_) : WidgetEditor(parent_, icon, name), soValue(soValue_) {
  so[0]=NULL; so[1]=NULL; so[2]=NULL;
  constructor(name);
}

Vec3fEditor::Vec3fEditor(QObject *parent_, const QIcon& icon, const string &name, SoSFFloat *soX_, SoSFFloat *soY_, SoSFFloat *soZ_) : WidgetEditor(parent_, icon, name), soValue(NULL) {
  so[0]=soX_; so[1]=soY_; so[2]=soZ_;
  constructor(name);
}

void Vec3fEditor::constructor(const string &name) {
  // add the label and a 3 spinboxes for the values
  layout->addWidget(new QLabel((name+":").c_str()), 0, 1);
  for(int i=0; i<3; i++) {
    spinBox[i]=new QDoubleSpinBox;
    spinBox[i]->setSingleStep(0.01);
    spinBox[i]->setRange(-DBL_MAX, DBL_MAX);
    spinBox[i]->setDecimals(6);
    spinBox[i]->setFixedWidth(QFontMetrics(spinBox[i]->font()).width("-888.888888000"));
    connect(spinBox[i], SIGNAL(valueChanged(double)), this, SLOT(valueChangedSlot()));
    layout->addWidget(spinBox[i], 0, i+2);
  }
}

void Vec3fEditor::valueChangedSlot() {
  // get values
  float x, y, z;
  x=spinBox[0]->value();
  y=spinBox[1]->value();
  z=spinBox[2]->value();
  // set so
  if(soValue)
    soValue->setValue(x, y, z);
  else {
    so[0]->setValue(x);
    so[1]->setValue(y);
    so[2]->setValue(z);
  }
  // set OpenMBV
  if(ombvSetter) ombvSetter(x, y, z);
  // trigger valueChanged signal of Editor
  emit valueChanged();
}





TransRotEditor::TransRotEditor(QObject *parent_, const QIcon& icon, const string &name, SoSFVec3f *soTranslation_, SoSFRotation *soRotation_) : WidgetEditor(parent_, icon, name), soTranslation(soTranslation_), soRotation(soRotation_) {
  // add the label and a 6 spinboxes for the 3 translations and 3 rotations
  layout->addWidget(new QLabel((name+":").c_str()), 0, 1);
  for(int i=0; i<3; i++) {
    spinBox[i  ]=new QDoubleSpinBox;
    spinBox[i+3]=new QDoubleSpinBox;
    spinBox[i  ]->setRange(-DBL_MAX, DBL_MAX);
    spinBox[i+3]->setRange(0, 360); // degree
    spinBox[i  ]->setSingleStep(0.01);
    spinBox[i+3]->setSingleStep(10);// degree
    spinBox[i  ]->setDecimals(6);
    spinBox[i+3]->setDecimals(4);
    spinBox[i  ]->setFixedWidth(QFontMetrics(spinBox[i  ]->font()).width("-888.888888000"));
    spinBox[i+3]->setFixedWidth(QFontMetrics(spinBox[i+3]->font()).width("-888.888888000"));
    spinBox[i+3]->setSuffix(QString::fromUtf8("\xc2\xb0")); // utf8 degree sign
    connect(spinBox[i  ], SIGNAL(valueChanged(double)), this, SLOT(valueChangedSlot()));
    connect(spinBox[i+3], SIGNAL(valueChanged(double)), this, SLOT(valueChangedSlot()));
    layout->addWidget(spinBox[i  ], 0, i+2);
    layout->addWidget(spinBox[i+3], 1, i+2);
  }
}

void TransRotEditor::setDragger(SoGroup *draggerParent) {
  // dragger for initial translation and rotation
  soDraggerSwitch=new SoSwitch;
  draggerParent->addChild(soDraggerSwitch);
  soDraggerSwitch->whichChild.setValue(SO_SWITCH_NONE);
  // the dragger is added on the first click on the entry in the property menu (delayed create)
  // (Because the constructor of the dragger takes a long time to execute (performace reason))
  soDragger=NULL;

  // create a action to active the Dragger
  draggerAction=new QAction(Utils::QIconCached(":/centerballdragger.svg"),(name+" Dragger").c_str(), this);
  draggerAction->setCheckable(true);
  draggerAction->setObjectName((string(parent->metaObject()->className())+"::"+name+".dragger").c_str());
  connect(draggerAction, SIGNAL(toggled(bool)), this, SLOT(draggerSlot(bool)));
}

void TransRotEditor::valueChangedSlot() {
  // get values
  float x, y, z, a, b, g;
  x=spinBox[0]->value();
  y=spinBox[1]->value();
  z=spinBox[2]->value();
  a=spinBox[3]->value()*M_PI/180;
  b=spinBox[4]->value()*M_PI/180;
  g=spinBox[5]->value()*M_PI/180;
  // set so
  soTranslation->setValue(x, y, z);
  *soRotation=Utils::cardan2Rotation(SbVec3f(a, b, g)).invert();
  // set OpenMBV
  if(ombvTransSetter && ombvRotSetter) {
    ombvTransSetter(x, y, z);
    ombvRotSetter(a, b, g);
  }
  // set dragger
  if(soDragger) {
    SbMatrix m;
    m.setTransform(soTranslation->getValue(), soRotation->getValue(), SbVec3f(1,1,1));
    soDragger->setMotionMatrix(m);
  }
  // trigger valueChanged signal of Editor
  emit valueChanged();
}

void TransRotEditor::draggerSlot(bool newValue) {
  if(soDragger==NULL) { // delayed create is now done
    soDragger=new SoCenterballDragger;
    soDraggerSwitch->addChild(soDragger);
    soDragger->addMotionCallback(draggerMoveCB, this);
    soDragger->addFinishCallback(draggerFinishedCB, this);
    // scale of the dragger
    SoSurroundScale *draggerScale=new SoSurroundScale;
    draggerScale->setDoingTranslations(false);
    draggerScale->numNodesUpToContainer.setValue(5);
    draggerScale->numNodesUpToReset.setValue(4);
    soDragger->setPart("surroundScale", draggerScale);
  }
  // switch the Dragger on of off depending on the Dragger action state
  if(newValue)
    soDraggerSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soDraggerSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

void TransRotEditor::draggerMoveCB(void *data, SoDragger *dragger_) {
  TransRotEditor *me=static_cast<TransRotEditor*>(data);
  SoCenterballDragger* dragger=(SoCenterballDragger*)dragger_;
  // get values
  SbVec3f t, s;
  SbRotation A, dummy;
  dragger->getMotionMatrix().getTransform(t, A, s, dummy);
  // set so
  me->soTranslation->setValue(t);
  me->soRotation->setValue(A);
  // set Qt
  for(int i=0; i<3; i++) {
    me->spinBox[i]->blockSignals(true);
    me->spinBox[i]->setValue(t[i]);
    me->spinBox[i]->blockSignals(false);
  }
  float v[3];
  Utils::rotation2Cardan(A.inverse()).getValue(v[0], v[1], v[2]);
  for(int i=0; i<3; i++) {
    me->spinBox[i+3]->blockSignals(true); // block signals to prevent endless loops
    if(v[i]<0) v[i]+=2*M_PI; // convert to positive angle (the spin box uses [0, 2*pi])
    me->spinBox[i+3]->setValue(v[i]*180/M_PI); // set the value
    me->spinBox[i+3]->blockSignals(false); // unblock all signals after setting the value
  }
  // set OpenMBV
  if(me->ombvTransSetter && me->ombvRotSetter) {
    me->ombvTransSetter(t[0], t[1], t[2]);
    me->ombvRotSetter(v[0], v[1], v[2]);
  }
  // show current trans/rot in status bar
  MainWindow::getInstance()->statusBar()->showMessage(QString("Trans: [%1, %2, %3]; Rot: [%4, %5, %6]").
    arg(t[0],0,'f',6).arg(t[1],0,'f',6).arg(t[2],0,'f',6).
    arg(v[0],0,'f',6).arg(v[1],0,'f',6).arg(v[2],0,'f',6));
  // trigger valueChanged signal of Editor
  emit me->valueChanged();
}

void TransRotEditor::draggerFinishedCB(void *data, SoDragger *dragger_) {
  TransRotEditor *me=static_cast<TransRotEditor*>(data);
  // print final trans/rot to stdout for an Object
  Object *obj=dynamic_cast<Object*>(me->parent);
  if(obj) {
    cout<<"New initial translation/rotation for: "<<obj->getPath()<<endl
        <<"Translation: ["<<me->spinBox[0]->value()<<", "<<me->spinBox[1]->value()<<", "<<me->spinBox[2]->value()<<"]"<<endl
        <<"Rotation: ["<<me->spinBox[3]->value()*M_PI/180<<", "<<me->spinBox[4]->value()*M_PI/180<<", "<<me->spinBox[5]->value()*M_PI/180<<"]"<<endl;
  }
  // trigger valueChanged signal of Editor
  emit me->valueChanged();
}
