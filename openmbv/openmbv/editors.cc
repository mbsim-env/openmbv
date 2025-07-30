#include "config.h"
#include <editors.h>
#include <Inventor/draggers/SoCenterballDragger.h>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QScrollArea>
#include <QPushButton>
#include <QApplication>
#include <QCheckBox>
#include <QMenu>
#include "utils.h"
#include <cfloat>
#include <mainwindow.h>
#include <objectfactory.h>
#include <compoundrigidbody.h>
#include <nurbsdisk.h>
#include <Inventor/nodes/SoSurroundScale.h>
#include <Inventor/fields/SoSFFloat.h>
#include <Inventor/fields/SoMFFloat.h>
#include <Inventor/fields/SoSFInt32.h>
#include <Inventor/fields/SoSFVec3f.h>
#include <Inventor/fields/SoSFRotation.h>

using namespace std;

namespace OpenMBVGUI {

PropertyDialog::PropertyDialog(QObject *parentObject_) : QDialog(MainWindow::getInstance()), parentObject(parentObject_) {
  // main layout
  mainLayout=new QGridLayout;
  setLayout(mainLayout);
  mainLayout->setColumnStretch(0, 1);

  // the layout is always 3 column width
  // Variant 1 for small widgets: i,0: Icon; i,1: Label; i,2: Widget
  // Variant 2 for large widgets: i,0: Icon; i,1: Label; i+1,1-2: Widget
  layout=new QGridLayout;
  layout->setColumnStretch(0, 0);
  layout->setColumnStretch(1, 0);
  layout->setColumnStretch(2, 1);
  scrollWidget=new QWidget;
  scrollWidget->setLayout(layout);
  scrollArea=new QScrollArea;
  scrollArea->setWidgetResizable(true);
  Utils::enableTouch(scrollArea);
  mainLayout->addWidget(scrollArea, 1, 0);

  // set window title
  setWindowTitle("OpenMBV Property Editor");

  contextMenu=new QMenu("Context Menu");
  contextMenu->setSeparatorsCollapsible(false);
  auto *dialogAction=new QAction("Properties...", this);
  contextMenu->addAction(dialogAction);
  connect(dialogAction, &QAction::triggered, this, &PropertyDialog::openDialogSlot);
  auto *sep2=new QAction(this);
  sep2->setSeparator("Convenience properties actions");
  contextMenu->addAction(sep2);

  auto *sep=new QAction(this);
  sep->setSeparator("Context actions");
  sep->setObjectName("PropertyDialog::AFTER_PROPERTIES__BEFORE_CONTEXT");
  contextMenu->addAction(sep);
  contextMenu->addSeparator();
}

PropertyDialog::~PropertyDialog() {
  for(auto & i : editor)
    delete i;
  contextMenu->deleteLater();
}

void PropertyDialog::openDialogSlot() {
  scrollArea->setWidget(scrollWidget); // must be called after all widgets are added to "layout"
  show();
}

void PropertyDialog::setParentObject(QObject *parentObject_) {
  parentObject=parentObject_;
}

void PropertyDialog::addPropertyAction(QAction *action) {
  for(int i=0; i<contextMenu->actions().size(); i++)
    if(contextMenu->actions()[i]->objectName()=="PropertyDialog::AFTER_PROPERTIES__BEFORE_CONTEXT") {
      contextMenu->insertAction(contextMenu->actions()[i], action);
      break;
    }
}

void PropertyDialog::addPropertyActionGroup(QActionGroup *actionGroup) {
  for(int j=0; j<actionGroup->actions().size(); j++)
    for(int i=0; i<contextMenu->actions().size(); i++)
      if(contextMenu->actions()[i]->objectName()=="PropertyDialog::AFTER_PROPERTIES__BEFORE_CONTEXT") {
        contextMenu->insertAction(contextMenu->actions()[i], actionGroup->actions()[j]);
        break;
      }
}

void PropertyDialog::addContextAction(QAction *action) {
  contextMenu->addAction(action);
}

void PropertyDialog::addSmallRow(const QIcon& icon, const std::string& name, QWidget *widget) {
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  QFontInfo fontinfo(font());
  iconLabel->setPixmap(icon.pixmap(fontinfo.pixelSize(),fontinfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel((name+":").c_str()), row, 1);
  layout->addWidget(widget, row, 2);
}

void PropertyDialog::addLargeRow(const QIcon& icon, const std::string& name, QWidget *widget) {
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  QFontInfo fontinfo(font());
  iconLabel->setPixmap(icon.pixmap(fontinfo.pixelSize(),fontinfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel((name+":").c_str()), row, 1, 1, 2);
  layout->addWidget(widget, row+1, 1, 1, 2);
}

void PropertyDialog::addSmallRow(const QIcon& icon, const std::string& name, QLayout *subLayout) {
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  QFontInfo fontinfo(font());
  iconLabel->setPixmap(icon.pixmap(fontinfo.pixelSize(),fontinfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel((name+":").c_str()), row, 1);
  layout->addLayout(subLayout, row, 2);
}

void PropertyDialog::addLargeRow(const QIcon& icon, const std::string& name, QLayout *subLayout) {
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  QFontInfo fontinfo(font());
  iconLabel->setPixmap(icon.pixmap(fontinfo.pixelSize(),fontinfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel((name+":").c_str()), row, 1, 1, 2);
  layout->addLayout(subLayout, row+1, 1, 1, 2);
}

void PropertyDialog::updateHeader() {
  auto *obj=dynamic_cast<Object*>(parentObject);
  if(obj) { // if it is a dialog for a Object
    auto *header=new QGridLayout;
    mainLayout->addLayout(header, 0, 0);
    header->setContentsMargins(0, 0, 0, 16);
    header->setColumnStretch(0, 0);
    header->setColumnStretch(1, 1);
    // display Object icon
    auto *objectIcon=new QLabel;
    QFontInfo fontinfo(font());
    objectIcon->setPixmap(static_cast<QTreeWidgetItem*>(obj)->icon(0).pixmap(fontinfo.pixelSize()*3,fontinfo.pixelSize()*3));
    header->addWidget(objectIcon, 0, 0, 2, 1);
    // diaplay Object name
    header->addWidget(new QLabel("<big><b>"+
      QString(obj->metaObject()->className()).replace("OpenMBVGUI::", "")+ // remove the namespace
      " XML Values of</b></big>"), 0, 1);
    // diaplay Object path
    auto *path=new QLabel(("<b>"+obj->getObject()->getFullName()+"</b>").c_str());
    path->setWordWrap(true);
    header->addWidget(path, 1, 1);
  }
}

void PropertyDialog::addEditor(Editor *child) {
  editor.push_back(child);
}

QList<QAction*> PropertyDialog::getActions() {
  QList<QAction*> list;
  for(auto & i : editor)
    list.append(i->findChildren<QAction*>());
  return list;
}





Editor::Editor(PropertyDialog *parent_, const QIcon &icon, const std::string &name) : dialog(parent_) {
  dialog->addEditor(this);
}

void Editor::getSelAndCur(QTreeWidgetItem *item, queue<bool> &sel, queue<bool> &cur) {
  sel.push(item->isSelected());
  item->setSelected(false);
  cur.push(item==MainWindow::getInstance()->objectList->currentItem());
}
void Editor::setSelAndCur(QTreeWidgetItem *item, queue<bool> &sel, queue<bool> &cur) {
  item->setSelected(sel.front());
  sel.pop();
  if(cur.front())
    MainWindow::getInstance()->objectList->setCurrentItem(item, 0, QItemSelectionModel::NoUpdate);
  cur.pop();
}
void Editor::unsetClone(Object *obj) {
  obj->clone=nullptr;
}
void Editor::replaceObject() {
  auto *obj=dynamic_cast<Object*>(dialog->getParentObject());
  if(!obj)
    return;
  QTreeWidget *objectList=MainWindow::getInstance()->objectList;

  // save selection and current item and clear selection under obj
  queue<bool> selected;
  queue<bool> current;
  Utils::visitTreeWidgetItems<QTreeWidgetItem*>(obj, [&selected, &current](auto && PH1) { return getSelAndCur(std::forward<decltype(PH1)>(PH1), selected, current); });

  // re-add this object using the same OpenMBVCppInterface::Object
  QTreeWidgetItem *treeWidgetParent=obj->QTreeWidgetItem::parent();
  int ind;
  QTreeWidgetItem *parentItem;
  SoSeparator *soParent;
  if(treeWidgetParent) {
    ind=treeWidgetParent->indexOfChild(static_cast<QTreeWidgetItem*>(obj));
    parentItem=treeWidgetParent;
    soParent=static_cast<Object*>(treeWidgetParent)->soSep;
  }
  else {
    ind=MainWindow::getInstance()->getRootItemIndexOfChild(static_cast<Group*>(obj));
    parentItem=objectList->invisibleRootItem();
    soParent=MainWindow::getInstance()->getSceneRoot();
  }
  Object *newObj=ObjectFactory::create(obj->object, parentItem, soParent, -ind-3); // we mark a clone object in the ctor by using idx=-ind-3 (since idx runs from -1, 0, 1, 2, ...)
  // delete this object (it is replaced by the above newly added)
  // but do not remove the OpenMBVCppInterface::Object
  obj->isCloneToBeDeleted=true;
  delete obj;
  Utils::visitTreeWidgetItems<Object*>(newObj, &unsetClone);
  // update the scene
  MainWindow::getInstance()->frame->touch();
  // apply object filter
  MainWindow::getInstance()->objectListFilter->applyFilter();
  // restore selection and current item
  Utils::visitTreeWidgetItems<QTreeWidgetItem*>(newObj, [&selected, &current](auto && PH1) { return setSelAndCur(std::forward<decltype(PH1)>(PH1), selected, current); });
  objectList->scrollToItem(objectList->currentItem());
}





BoolEditor::BoolEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name, const std::string &qtObjectName,
                       bool replaceObjOnChange_) : Editor(parent_, icon, name),replaceObjOnChange(replaceObjOnChange_) {
  checkbox=new QCheckBox;
  checkbox->installEventFilter(&IgnoreWheelEventFilter::instance);
  connect(checkbox, &QCheckBox::stateChanged, this, &BoolEditor::valueChangedSlot);
  dialog->addSmallRow(icon, name, checkbox);

  action=new QAction(icon, name.c_str(), this);
  action->setCheckable(true);
  action->setObjectName(qtObjectName.c_str());
  connect(action,&QAction::changed,this,&BoolEditor::actionChangedSlot);
}

void BoolEditor::valueChangedSlot(int state) {
  if(ombvSetter) ombvSetter(state==Qt::Checked); // set OpenMBV
  action->blockSignals(true);
  action->setChecked(state==Qt::Checked);
  action->blockSignals(false);
  if(replaceObjOnChange)
    replaceObject();
  stateChanged(state==Qt::Checked);
}

void BoolEditor::actionChangedSlot() {
  checkbox->setCheckState(action->isChecked()?Qt::Checked:Qt::Unchecked);
}





FloatEditor::FloatEditor(PropertyDialog *parent_, const QIcon& icon, const string &name) : Editor(parent_, icon, name) {
  // add the label and a spinbox for the value
  factor=1;
  spinBox=new QDoubleSpinBox;
  spinBox->installEventFilter(&IgnoreWheelEventFilter::instance);
  spinBox->setSingleStep(0.01);
  spinBox->setRange(-DBL_MAX, DBL_MAX);
  spinBox->setDecimals(6);
  spinBox->setMinimumWidth(QFontMetrics(spinBox->font()).
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
    horizontalAdvance
#else
    width
#endif
    ("-888.888888000"));
  connect(spinBox, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &FloatEditor::valueChangedSlot);
  dialog->addSmallRow(icon, name, spinBox);
}

void FloatEditor::valueChangedSlot(double newValue) {
  if(ombvSetter)
  {
    if(spinBox->specialValueText()=="" || newValue!=spinBox->minimum())
      ombvSetter(newValue*factor); // set OpenMBV
    else
      ombvSetter(numeric_limits<double>::quiet_NaN()); // set OpenMBV
  }
  replaceObject();
}





FloatMatrixEditor::FloatMatrixEditor(PropertyDialog *parent_, const QIcon& icon, const string &name, unsigned int rows_, unsigned int cols_) : Editor(parent_, icon, name) {
  auto *layout=new QGridLayout;

  rows=rows_;
  cols=cols_;
  QPushButton *button;
  int layoutCols=0;

  // if rows==0 add a add and remove row button
  if(rows==0) {
    button=new QPushButton("Add row");
    connect(button, &QPushButton::released, this, &FloatMatrixEditor::addRowSlot);
    layout->addWidget(button, 0, layoutCols++);

    button=new QPushButton("Remove row");
    connect(button, &QPushButton::released, this, &FloatMatrixEditor::removeRowSlot);
    layout->addWidget(button, 0, layoutCols++);
  }
  // if cols==0 add a add and remove col button
  if(cols==0) {
    button=new QPushButton("Add column");
    connect(button, &QPushButton::released, this, &FloatMatrixEditor::addColumnSlot);
    layout->addWidget(button, 0, layoutCols++);

    button=new QPushButton("Remove column");
    connect(button, &QPushButton::released, this, &FloatMatrixEditor::removeColumnSlot);
    layout->addWidget(button, 0, layoutCols++);
  }

  table=new QTableWidget(rows, cols);
  table->installEventFilter(&IgnoreWheelEventFilter::instance);
  layout->addWidget(table, 1, 0, 1, layoutCols);

  dialog->addLargeRow(icon, name, layout);
}

void FloatMatrixEditor::addRow() {
  table->setRowCount(table->rowCount()+1);
  for(int c=0; c<table->columnCount(); c++) {
    auto *cell=new QDoubleSpinBox;
    cell->installEventFilter(&IgnoreWheelEventFilter::instance);
    table->setCellWidget(table->rowCount()-1, c, cell);
    cell->setSingleStep(0.01);
    cell->setRange(-DBL_MAX, DBL_MAX);
    cell->setDecimals(6);
    cell->setMinimumWidth(QFontMetrics(cell->font()).
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
      horizontalAdvance
#else
      width
#endif
      ("-888.888888000"));
    connect(cell, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &FloatMatrixEditor::valueChangedSlot);
  }
}

void FloatMatrixEditor::addRowSlot() {
  addRow();
  valueChangedSlot();
}

void FloatMatrixEditor::removeRowSlot() {
  table->setRowCount(table->rowCount()-1);
  valueChangedSlot();
}

void FloatMatrixEditor::addColumn() {
  table->setColumnCount(table->columnCount()+1);
  for(int r=0; r<table->rowCount(); r++) {
    auto *cell=new QDoubleSpinBox;
    cell->installEventFilter(&IgnoreWheelEventFilter::instance);
    table->setCellWidget(r, table->columnCount()-1, cell);
    cell->setSingleStep(0.01);
    cell->setRange(-DBL_MAX, DBL_MAX);
    cell->setDecimals(6);
    cell->setMinimumWidth(QFontMetrics(cell->font()).
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
      horizontalAdvance
#else
      width
#endif
      ("-888.888888000"));
    connect(cell, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &FloatMatrixEditor::valueChangedSlot);
  }
}

void FloatMatrixEditor::addColumnSlot() {
  addColumn();
  valueChangedSlot();
}

void FloatMatrixEditor::removeColumnSlot() {
  table->setColumnCount(table->columnCount()-1);
  valueChangedSlot();
}

void FloatMatrixEditor::valueChangedSlot() {
  // set OpenMBV if std::vector
  if(ombvSetterVector && ombvGetterVector) {
    // asserts
    assert(cols==1 || rows==1);
    std::vector<double> vec;
    // get values from Qt
    vec.reserve((cols==1?table->rowCount():table->columnCount()));
    for(int i=0; i<(cols==1?table->rowCount():table->columnCount()); i++)
      vec.push_back(static_cast<QDoubleSpinBox*>(table->cellWidget(cols==1?i:0, cols==1?0:i))->value());
    // set values to OpenMBV
    ombvSetterVector(vec);
  }
  // set OpenMBV if std::vector<std::vector>
  if(ombvSetterMatrix && ombvGetterMatrix) {
    // asserts
    std::vector<std::vector<double> > mat;
    // get values from Qt
    for(int r=0; r<table->rowCount(); r++) {
      std::vector<double> row;
      row.reserve(table->columnCount());
for(int c=0; c<table->columnCount(); c++)
        row.push_back(static_cast<QDoubleSpinBox*>(table->cellWidget(r, c))->value());
      mat.push_back(row);
    }
    // set values to OpenMBV
    ombvSetterMatrix(mat);
  }
  // set OpenMBV if PolygonPoint
  if(ombvSetterPolygonPoint && ombvGetterPolygonPoint) {
    // asserts
    assert(cols==3);
    shared_ptr<vector<shared_ptr<OpenMBV::PolygonPoint> > > contour;
    // create new heap PolygonPoints
    contour=make_shared<vector<shared_ptr<OpenMBV::PolygonPoint> > >();
    // get values from Qt
    for(int r=0; r<table->rowCount(); r++) {
      shared_ptr<OpenMBV::PolygonPoint> polyPoint=OpenMBV::PolygonPoint::create(
                               static_cast<QDoubleSpinBox*>(table->cellWidget(r, 0))->value(),
                               static_cast<QDoubleSpinBox*>(table->cellWidget(r, 1))->value(),
        static_cast<int>(round(static_cast<QDoubleSpinBox*>(table->cellWidget(r, 2))->value())));
      contour->push_back(polyPoint);
    }
    // set values to OpenMBV
    ombvSetterPolygonPoint(contour);
  }

  replaceObject();
}





IntEditor::IntEditor(PropertyDialog *parent_, const QIcon& icon, const string &name) : Editor(parent_, icon, name) {
  // add the label and a spinbox for the value
  spinBox=new QSpinBox;
  spinBox->installEventFilter(&IgnoreWheelEventFilter::instance);
  spinBox->setRange(INT_MIN, INT_MAX);
  spinBox->setMinimumWidth(QFontMetrics(spinBox->font()).
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
    horizontalAdvance
#else
    width
#endif
    ("-888.888888000"));
  connect(spinBox, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &IntEditor::valueChangedSlot);
  dialog->addSmallRow(icon, name, spinBox);
}

void IntEditor::valueChangedSlot(int newValue) {
  if(ombvSetter)
    ombvSetter(newValue); // set OpenMBV
  replaceObject();
}





StringEditor::StringEditor(PropertyDialog *parent_, const QIcon& icon, const string &name) : Editor(parent_, icon, name) {
  // add the label and a lineEdit for the value
  lineEdit=new QLineEdit;
  lineEdit->installEventFilter(&IgnoreWheelEventFilter::instance);
  connect(lineEdit, &QLineEdit::textChanged, this, &StringEditor::valueChangedSlot);
  dialog->addSmallRow(icon, name, lineEdit);
}

void StringEditor::valueChangedSlot(const QString &text) {
  if(ombvSetter)
    ombvSetter(text.toStdString()); // set OpenMBV
  replaceObject();
}





ComboBoxEditor::ComboBoxEditor(PropertyDialog *parent_, const QIcon& icon, const string &name,
  const std::vector<std::tuple<int, string, QIcon, string> > &list) : Editor(parent_, icon, name) {

  // add the label and a comboBox for the value
  comboBox=new QComboBox;
  comboBox->installEventFilter(&IgnoreWheelEventFilter::instance);
  for(const auto & i : list)
    comboBox->addItem(get<2>(i), get<1>(i).c_str(), QVariant(get<0>(i)));
  connect(comboBox, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &ComboBoxEditor::valueChangedSlot);
  dialog->addSmallRow(icon, name, comboBox);

  actionGroup=new QActionGroup(this);
  auto *sep1=new QAction(actionGroup);
  sep1->setSeparator(name.c_str());
  actionGroup->addAction(sep1);
  for(size_t i=0; i<list.size(); i++) {
    auto *action=new QAction(get<2>(list[i]), get<1>(list[i]).c_str(), actionGroup);
    action->setObjectName(get<3>(list[i]).c_str());
    action->setData(QVariant(static_cast<int>(i)));
    action->setCheckable(true);
    actionGroup->addAction(action);
  }
  auto *sep=new QAction(actionGroup);
  sep->setSeparator("");
  actionGroup->addAction(sep);
  connect(actionGroup,&QActionGroup::triggered,this,&ComboBoxEditor::actionChangedSlot);
}

void ComboBoxEditor::valueChangedSlot(int newValue) {
  if(ombvSetter) ombvSetter(comboBox->itemData(newValue).toInt()); // set OpenMBV
  for(int i=1; i<actionGroup->actions().size()-1; i++) {
    actionGroup->actions()[i]->blockSignals(true);
    actionGroup->actions()[i]->setChecked(i-1==comboBox->itemData(newValue).toInt());
    actionGroup->actions()[i]->blockSignals(false);
  }
  replaceObject();
}

void ComboBoxEditor::actionChangedSlot(QAction* action) {
  comboBox->setCurrentIndex(action->data().toInt());
}





Vec3fEditor::Vec3fEditor(PropertyDialog *parent_, const QIcon& icon, const string &name) : Editor(parent_, icon, name) {
  // add the label and a 3 spinboxes for the values
  auto *box=new QHBoxLayout;
  for(auto & i : spinBox) {
    i=new QDoubleSpinBox;
    i->installEventFilter(&IgnoreWheelEventFilter::instance);
    i->setSingleStep(0.01);
    i->setRange(-DBL_MAX, DBL_MAX);
    i->setDecimals(6);
    i->setMinimumWidth(QFontMetrics(i->font()).
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
      horizontalAdvance
#else
      width
#endif
      ("-888.888888000"));
    connect(i, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &Vec3fEditor::valueChangedSlot);
    box->addWidget(i);
  }
  dialog->addSmallRow(icon, name, box);
}

void Vec3fEditor::valueChangedSlot() {
  // set OpenMBV
  if(ombvSetter) ombvSetter(spinBox[0]->value(), spinBox[1]->value(), spinBox[2]->value());
  replaceObject();
}



ColorEditor::ColorEditor(PropertyDialog *parent_, const QIcon& icon, const string &name, bool showResetHueButton) : Editor(parent_, icon, name) {
  auto *box=new QHBoxLayout;
  colorDialog=new QColorDialog();
  colorDialog->setOption(QColorDialog::NoButtons);
  connect(colorDialog, &QColorDialog::currentColorChanged, this, &ColorEditor::valueChangedSlot);
  auto *showDL=new QPushButton("Color...");
  connect(showDL, &QPushButton::clicked, this, &ColorEditor::showDialog);
  box->addWidget(showDL);
  if(showResetHueButton) {
    auto *resetHue=new QPushButton("Reset to hue from HDF5");
    connect(resetHue, &QPushButton::clicked, this, &ColorEditor::resetHue);
    box->addWidget(resetHue);
  }
  dialog->addSmallRow(icon, name, box);
}

void ColorEditor::valueChangedSlot() {
  // set OpenMBV
  if(ombvSetter) {
    qreal h, s, v;
    colorDialog->currentColor().getHsvF(&h, &s, &v);
    ombvSetter(h, s, v);
  }
  replaceObject();
}

void ColorEditor::showDialog() {
  colorDialog->open();
}

void ColorEditor::resetHue() {
  // set OpenMBV (only hue part)
  if(ombvSetter) {
    qreal h, s, v;
    colorDialog->currentColor().getHsvF(&h, &s, &v);
    h=-1;
    ombvSetter(h, s, v);
  }
  replaceObject();
}





TransRotEditor::TransRotEditor(PropertyDialog *parent_, const QIcon& icon, const string &name) : Editor(parent_, icon, name) {
  // add the label and a 6 spinboxes for the 3 translations and 3 rotations
  auto *trans=new QHBoxLayout;
  auto *rot=new QHBoxLayout;

  // create a action to active the Dragger
  draggerCheckBox=new QCheckBox;
  dialog->addSmallRow(Utils::QIconCached("centerballdragger.svg"), name+" dragger", draggerCheckBox);
  connect(draggerCheckBox, &QCheckBox::stateChanged, this, &TransRotEditor::draggerSlot);

  for(int i=0; i<3; i++) {
    spinBox[i  ]=new QDoubleSpinBox;
    spinBox[i+3]=new QDoubleSpinBox;
    spinBox[i  ]->installEventFilter(&IgnoreWheelEventFilter::instance);
    spinBox[i+3]->installEventFilter(&IgnoreWheelEventFilter::instance);
    spinBox[i  ]->setRange(-DBL_MAX, DBL_MAX);
    spinBox[i+3]->setRange(-DBL_MAX, DBL_MAX); // degree
    spinBox[i  ]->setSingleStep(0.01);
    spinBox[i+3]->setSingleStep(10);// degree
    spinBox[i  ]->setDecimals(6);
    spinBox[i+3]->setDecimals(6);
    spinBox[i  ]->setMinimumWidth(QFontMetrics(spinBox[i  ]->font()).
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
      horizontalAdvance
#else
      width
#endif
      ("-888.888888000"));
    spinBox[i+3]->setMinimumWidth(QFontMetrics(spinBox[i+3]->font()).
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
      horizontalAdvance
#else
      width
#endif
      ("-888.888888000"));
    spinBox[i+3]->setSuffix(QString::fromUtf8(R"(°)")); // utf8 degree sign
    connect(spinBox[i  ], static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &TransRotEditor::valueChangedSlot);
    connect(spinBox[i+3], static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &TransRotEditor::valueChangedSlot);
    trans->addWidget(spinBox[i]);
    rot->addWidget(spinBox[i+3]);
  }
  dialog->addSmallRow(icon, name+" translation", trans);
  dialog->addSmallRow(icon, name+" rotation", rot);

  // dragger for initial translation and rotation
  soDraggerSwitch=new SoSwitch;
  soDraggerSwitch->ref();
  soDraggerSwitch->whichChild.setValue(SO_SWITCH_NONE);
  // the dragger is added on the first click on the entry in the property menu (delayed create)
  // (Because the constructor of the dragger takes a long time to execute (performace reason))
  soDragger=nullptr;

  soTranslation=new SoTranslation;
  soTranslation->ref();
  soRotation=new SoRotation;
  soRotation->ref();
}

TransRotEditor::~TransRotEditor() {
  soDraggerSwitch->unref();
  soTranslation->unref();
  soRotation->unref();
}

void TransRotEditor::setGroupMembers(SoGroup *grp) {
  grp->removeAllChildren();
  grp->addChild(soDraggerSwitch);
  grp->addChild(soTranslation);
  grp->addChild(soRotation);
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
  soTranslation->translation.setValue(x, y, z);
  soRotation->rotation=Utils::cardan2Rotation(SbVec3f(a, b, g)).invert();
  // set OpenMBV
  if(ombvTransSetter && ombvRotSetter) {
    ombvTransSetter(x, y, z);
    ombvRotSetter(a, b, g);
  }
  // set dragger
  if(soDragger) {
    SbMatrix m;
    m.setTransform(soTranslation->translation.getValue(), soRotation->rotation.getValue(), SbVec3f(1,1,1));
    soDragger->setMotionMatrix(m);
  }

  // do NOT replace the object here it is update by the SoFields
}

void TransRotEditor::draggerSlot(int state) {
  if(soDragger==nullptr) { // delayed create is now done
    soDragger=new SoCenterballDragger;
    soDraggerSwitch->addChild(soDragger);
    soDragger->addMotionCallback(draggerMoveCB, this);
    soDragger->addFinishCallback(draggerFinishedCB, this);
    // scale of the dragger
    auto *draggerScale=new SoSurroundScale;
    draggerScale->setDoingTranslations(false);
    draggerScale->numNodesUpToContainer.setValue(5);
    draggerScale->numNodesUpToReset.setValue(4);
    soDragger->setPart("surroundScale", draggerScale);
    SbMatrix m;
    m.setTransform(soTranslation->translation.getValue(), soRotation->rotation.getValue(), SbVec3f(1,1,1));
    soDragger->setMotionMatrix(m);
  }
  // switch the Dragger on of off depending on the Dragger action state
  if(state==Qt::Checked)
    soDraggerSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soDraggerSwitch->whichChild.setValue(SO_SWITCH_NONE);
  if(ombvDraggerSetter) ombvDraggerSetter(state==Qt::Checked); // set OpenMBV
}

void TransRotEditor::draggerMoveCB(void *data, SoDragger *dragger_) {
  auto *me=static_cast<TransRotEditor*>(data);
  auto* dragger=(SoCenterballDragger*)dragger_;
  // get values
  SbVec3f t, s;
  SbRotation A, dummy;
  dragger->getMotionMatrix().getTransform(t, A, s, dummy);
  // set so
  me->soTranslation->translation.setValue(t);
  me->soRotation->rotation.setValue(A);
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
}

void TransRotEditor::draggerFinishedCB(void *data, SoDragger *dragger_) {
  auto *me=static_cast<TransRotEditor*>(data);
  // print final trans/rot to stdout for an Object
  auto *obj=dynamic_cast<Object*>(me->dialog->parentObject);
  if(obj) {
    QString str("New translation [%1, %2, %3], rotation[%4, %5, %6] on %7");
    str=str.arg(me->spinBox[0]->value())
           .arg(me->spinBox[1]->value())
           .arg(me->spinBox[2]->value())
           .arg(me->spinBox[3]->value()*M_PI/180)
           .arg(me->spinBox[4]->value()*M_PI/180)
           .arg(me->spinBox[5]->value()*M_PI/180)
           .arg(obj->getObject()->getFullName().c_str());
    MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
    me->msg(Status)<<str.toStdString()<<endl;
  }
}





NotAvailableEditor::NotAvailableEditor(PropertyDialog *parent_, const QIcon &icon, const std::string &name) : Editor(parent_, icon, name) {
  dialog->addSmallRow(icon, name, new QLabel("Sorry, a editor for this value is not available.<br/>"
                                             "Please change this value manually in the XML-file."));
}

void PropertyDialog::closeEvent(QCloseEvent *event) {
  appSettings->set(AppSettings::propertydialog_geometry, saveGeometry());
  QDialog::closeEvent(event);
}

void PropertyDialog::showEvent(QShowEvent *event) {
  restoreGeometry(appSettings->get<QByteArray>(AppSettings::propertydialog_geometry));
  QDialog::showEvent(event);
}

}
