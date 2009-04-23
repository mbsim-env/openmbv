#include "config.h"
#include "objbody.h"
#include <QFile>
#include <QTextStream>
#include <tinyxml/tinynamespace.h>
#include <Inventor/nodes/SoShapeHints.h>

using namespace std;

double ObjBody::eps;

ObjBody::ObjBody(TiXmlElement *element, H5::Group *h5Parent) : RigidBody(element, h5Parent) {
  iconFile=":/objbody.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"objFileName");
  string fileName=e->GetText();
  // fix relative path name of file to be included (will hopefully work also on windows)
  fileName=fixPath(e->GetDocument()->ValueStr(), fileName);
  e=e->NextSiblingElement();
  bool textureFromFile=(e->GetText()==string("true"))?true:false;
  e=e->NextSiblingElement();
  bool materialFromFile=(e->GetText()==string("true"))?true:false;
  e=e->NextSiblingElement();
  string normals_=e->GetText();
  Normals normals;
  if(normals_=="fromObjFile") normals=fromObjFile;
  else if(normals_=="flat") normals=flat;
  else if(normals_=="smooth") normals=smooth;
  else normals=smoothIfLessBarrier;
  e=e->NextSiblingElement();
  double epsVertex=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  double epsNormal=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  smoothBarrier=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  string outline_=e->GetText();
  Outline outline;
  if(outline_=="none") outline=none;
  else if(outline_=="calculate") outline=calculate;
  else outline=fromFile;

  // create so
  // read obj
  v=new SoCoordinate3; v->point.deleteValues(0);
  t=new SoTextureCoordinate2; t->point.deleteValues(0);
  n=new SoNormal; n->vector.deleteValues(0);
  map<QString, SoMaterial*> material;
  map<QString, SoTexture2*> map_;
  QFile objFile(fileName.c_str());
  objFile.open(QIODevice::ReadOnly | QIODevice::Text);
  map<QString, MtlMapGroup> group;
  MtlMapGroup* curGroup=&group["OpenMBVDefaultMtlMap"];
  QString dummyStr;
  QRegExp commentRE("[ \t]*#.*");
  QRegExp emptyRE("[ \t]*");
  QRegExp usemtlRE("[ \t]*usemtl[ \t].*");
  QRegExp mtllibRE("[ \t]*mtllib[ \t].*");
  QRegExp usemapRE("[ \t]*usemap[ \t].*");
  QRegExp maplibRE("[ \t]*maplib[ \t].*");
  QRegExp vRE("[ \t]*v[ \t].*");
  QRegExp vtRE("[ \t]*vt[ \t].*");
  QRegExp vnRE("[ \t]*vn[ \t].*");
  QRegExp fRE("[ \t]*f[ \t].*");
  QRegExp fSplitRE("^([0-9]+)(?:/([0-9]*)(?:/([0-9]*))?)?$");
  QRegExp lRE("[ \t]*l[ \t].*");
  QRegExp lSplitRE("^([0-9]+)(?:/([0-9]*))?$");
  while(!objFile.atEnd()) {
    QByteArray line=objFile.readLine();
    line.resize(line.size()-1); // delete "\n"
    if(commentRE.exactMatch(line)) continue; // don't process comment lines
    if(emptyRE.exactMatch(line)) continue; // don't process empty lines
    if(materialFromFile && mtllibRE.exactMatch(line)) { // read material
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      string mtlFile=fixPath(fileName, dummyStr.toStdString());
      readMtlLib(mtlFile, material);
      continue;
    }
    if(textureFromFile && maplibRE.exactMatch(line)) { // read map
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      string mapFile=fixPath(fileName, dummyStr.toStdString());
      readMapLib(mapFile, map_);
      continue;
    }
    if(usemtlRE.exactMatch(line)) { // set current group
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      SoTexture2 *mapSave=curGroup->map; // save map
      curGroup=&group[dummyStr];
      if(materialFromFile) curGroup->mat=material[dummyStr];
      curGroup->map=mapSave; // set saved map
      continue;
    }
    if(usemapRE.exactMatch(line)) { // set current group
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      SoMaterial *matSave=curGroup->mat; // save mat
      curGroup=&group[dummyStr];
      if(textureFromFile) curGroup->map=map_[dummyStr];
      curGroup->mat=matSave; // set saved mat
      continue;
    }
    if(vRE.exactMatch(line)) { // read v
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>x>>y>>z;
      v->point.set1Value(v->point.getNum(), x,y,z);
      continue;
    }
    if(textureFromFile && vtRE.exactMatch(line)) { // read vt
      QTextStream stream(line);
      double x, y;
      stream>>dummyStr>>x>>y;
      t->point.set1Value(t->point.getNum(), x,y);
      continue;
    }
    if(vnRE.exactMatch(line)) { // read vn
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>x>>y>>z;
      n->vector.set1Value(n->vector.getNum(), x,y,z);
      continue;
    }
    if(fRE.exactMatch(line)) { // read f (write to current group)
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      fSplitRE.indexIn(dummyStr);
      int vi1=fSplitRE.cap(1).toInt()-1, ni1=fSplitRE.cap(3).toInt()-1;
      stream>>dummyStr;
      fSplitRE.indexIn(dummyStr);
      int vi2=fSplitRE.cap(1).toInt()-1, ni2=fSplitRE.cap(3).toInt()-1;
      while(!stream.atEnd()) {
        stream>>dummyStr;
        if(dummyStr=="") break;
        fSplitRE.indexIn(dummyStr);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi1);
        curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni1);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi2);
        curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni2);
        vi2=fSplitRE.cap(1).toInt()-1; ni2=fSplitRE.cap(3).toInt()-1;
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi2);
        curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni2);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), -1);
        curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), -1);
      }
      continue;
    }
    if(lRE.exactMatch(line)) { // read l (write to current group)
      QTextStream stream(line);
      stream>>dummyStr;
      while(!stream.atEnd()) {
        stream>>dummyStr;
        if(dummyStr=="") break;
        lSplitRE.indexIn(dummyStr);
        curGroup->ol->coordIndex.set1Value(curGroup->ol->coordIndex.getNum(), lSplitRE.cap(1).toInt()-1);
      }
      curGroup->ol->coordIndex.set1Value(curGroup->ol->coordIndex.getNum(), -1);
      continue;
    }
  }

  // create so
  map<QString, MtlMapGroup>::iterator i;
  if(epsVertex>0) { // combine vertex
   SoMFVec3f newvv;
   SoMFInt32 newvi;
   eps=epsVertex;
   combine(v->point, newvv, newvi);
   v->point.copyFrom(newvv);
   for(i=group.begin(); i!=group.end(); i++)
     convertIndex(i->second.f->coordIndex, newvi);
  }
  soSep->addChild(v);
  if(textureFromFile) soSep->addChild(t);
  if(normals==flat) { // flat normals
    SoNormalBinding *nb=new SoNormalBinding;
    soSep->addChild(nb);
    nb->value.setValue(SoNormalBinding::PER_FACE);
    for(i=group.begin(); i!=group.end(); i++) {
      if(i->second.mat) soSep->addChild(i->second.mat);
      if(i->second.map) soSep->addChild(i->second.map);
      soSep->addChild(i->second.f);
    }
  }
  else if(normals==fromObjFile) { // use normals form obj
    if(epsNormal>0) { // combine normals
     SoMFVec3f newvv;
     SoMFInt32 newvi;
     eps=epsNormal;
     combine(n->vector, newvv, newvi);
     n->vector.copyFrom(newvv);
     for(i=group.begin(); i!=group.end(); i++)
       convertIndex(i->second.f->normalIndex, newvi);
    }
    soSep->addChild(n);
    for(i=group.begin(); i!=group.end(); i++) {
      if(i->second.mat) soSep->addChild(i->second.mat);
      if(i->second.map) soSep->addChild(i->second.map);
      soSep->addChild(i->second.f);
    }
  }
  else { // smooth or real normals
    if(normals==smooth) smoothBarrier=M_PI; // if smooth simply use high value for barrier
    for(i=group.begin(); i!=group.end(); i++) {
      SoMFInt32 fni;
      SoMFVec3f n;
      SoMFInt32 oli;
      computeNormals(i->second.f->coordIndex, v->point, fni, n, oli);
      i->second.f->normalIndex.copyFrom(fni);
      if(epsNormal>0) { // combine normals
       SoMFVec3f newvv;
       SoMFInt32 newvi;
       eps=epsNormal;
       combine(n, newvv, newvi);
       i->second.n->vector.copyFrom(newvv);
       convertIndex(i->second.f->normalIndex, newvi);
      }
      soSep->addChild(i->second.n);
      if(i->second.mat) soSep->addChild(i->second.mat);
      if(i->second.map) soSep->addChild(i->second.map);
      soSep->addChild(i->second.f);
      if(outline==calculate) {
        i->second.ol->coordIndex.copyFrom(oli);
        soOutLineSep->addChild(i->second.ol);
      }
    }
  }
  if(outline==fromFile)
    for(i=group.begin(); i!=group.end(); i++)
      soOutLineSep->addChild(i->second.ol);
  soSep->addChild(soOutLineSwitch);
}

void ObjBody::readMtlLib(const std::string& mtlFile_, std::map<QString, SoMaterial*>& material) {
  QFile mtlFile(mtlFile_.c_str());
  mtlFile.open(QIODevice::ReadOnly | QIODevice::Text);
  QString dummyStr;
  QRegExp commentRE("[ \t]*#.*");
  QRegExp emptyRE("[ \t]*");
  QRegExp newmtlRE("[ \t]*newmtl[ \t].*");
  QRegExp KaRE("[ \t]*Ka[ \t].*");
  QRegExp KdRE("[ \t]*Kd[ \t].*");
  QRegExp KsRE("[ \t]*Ks[ \t].*");
  QRegExp NsRE("[ \t]*Ns[ \t].*");
  QRegExp dRE("[ \t]*d[ \t].*");
  SoMaterial *curMaterial;
  while(!mtlFile.atEnd()) {
    QByteArray line=mtlFile.readLine();
    line.resize(line.size()-1); // delete "\n"
    if(commentRE.exactMatch(line)) continue; // don't process comment lines
    if(emptyRE.exactMatch(line)) continue; // don't process empty lines
    if(newmtlRE.exactMatch(line)) { // new material
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      curMaterial=new SoMaterial;
      material[dummyStr]=curMaterial;
      continue;
    }
    if(KaRE.exactMatch(line)) { // set Ka
      QTextStream stream(line);
      double r, g, b;
      stream>>dummyStr>>r>>g>>b;
      curMaterial->ambientColor.setValue(r,g,b);
      continue;
    }
    if(KdRE.exactMatch(line)) { // set Kd
      QTextStream stream(line);
      double r, g, b;
      stream>>dummyStr>>r>>g>>b;
      curMaterial->diffuseColor.setValue(r,g,b);
      continue;
    }
    if(KsRE.exactMatch(line)) { // set Ks
      QTextStream stream(line);
      double r, g, b;
      stream>>dummyStr>>r>>g>>b;
      curMaterial->specularColor.setValue(r,g,b);
      continue;
    }
    if(NsRE.exactMatch(line)) { // set Ns
      QTextStream stream(line);
      double s;
      stream>>dummyStr>>s;
      curMaterial->shininess.setValue(s/1000);
      continue;
    }
    if(dRE.exactMatch(line)) { // set d
      QTextStream stream(line);
      double s;
      stream>>dummyStr>>s;
      curMaterial->transparency.setValue(1-s);
      continue;
    }
  }
}

void ObjBody::readMapLib(const std::string& mapFile_, std::map<QString, SoTexture2*>& map_) {
  QFile mapFile(mapFile_.c_str());
  mapFile.open(QIODevice::ReadOnly | QIODevice::Text);
  QString dummyStr;
  QRegExp commentRE("[ \t]*#.*");
  QRegExp emptyRE("[ \t]*");
  QRegExp newmapRE("[ \t]*newmap[ \t].*");
  QRegExp KdRE("[ \t]*Kd[ \t].*");
  SoTexture2 *curMap;
  while(!mapFile.atEnd()) {
    QByteArray line=mapFile.readLine();
    line.resize(line.size()-1); // delete "\n"
    if(commentRE.exactMatch(line)) continue; // don't process comment lines
    if(emptyRE.exactMatch(line)) continue; // don't process empty lines
    if(newmapRE.exactMatch(line)) { // new map
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      curMap=new SoTexture2;
      map_[dummyStr]=curMap;
      continue;
    }
    if(KdRE.exactMatch(line)) { // set Kd
      QTextStream stream(line);
      double r, g, b;
      stream>>dummyStr>>dummyStr;
      string imgName=fixPath(mapFile_, dummyStr.toStdString());
      QImage img(imgName.c_str());
      unsigned char *buf=new unsigned char[img.height()*img.width()*3];
      for(int y=0; y<img.height(); y++)
        for(int x=0; x<img.width(); x++) {
          int o=((img.height()-y-1)+img.height()*(img.width()-x-1))*3;
          QRgb rgba=img.pixel(x,y);
          buf[o+0]=qRed(rgba);
          buf[o+1]=qGreen(rgba);
          buf[o+2]=qBlue(rgba);
        }
      curMap->image.setValue(SbVec2s(img.height(),img.width()),3,buf);
      delete[]buf;
      continue;
    }
  }
}

ObjBody::MtlMapGroup::MtlMapGroup() {
  f=new SoIndexedFaceSet; f->ref(); f->coordIndex.deleteValues(0); f->normalIndex.deleteValues(0);
  ol=new SoIndexedLineSet; ol->ref(); ol->coordIndex.deleteValues(0);
  n=new SoNormal; n->ref(); n->vector.deleteValues(0);
  mat=0;
  map=0;
}

bool ObjBody::SbVec3fHash::operator()(const SbVec3f& v1, const SbVec3f& v2) const {
  float x1,y1,z1,x2,y2,z2;
  v1.getValue(x1,y1,z1);
  v2.getValue(x2,y2,z2);
  if(x1<x2-eps) return true;
  else if(x1>x2+eps) return false;
  else
    if(y1<y2-eps) return true;
    else if(y1>y2+eps) return false;
    else
      if(z1<z2-eps) return true;
      else if(z1>z2+eps) return false;
      else return false;
};

void ObjBody::combine(const SoMFVec3f& v, SoMFVec3f& newv, SoMFInt32& newvi) {
  map<SbVec3f, int, SbVec3fHash> hash;
  for(int i=0; i<v.getNum(); i++) {
    int &r=hash[*(v.getValues(i))]; // add vertex to a hash map
    // if vertex not exist in hash map copy it to newvv,
    // set corospondenting newvi,
    // and set hash map value to the index of this vertex
    if(r==0) {
      newv.set1Value(newv.getNum(), *v.getValues(i));
      newvi.set1Value(i, newv.getNum()-1);
      r=newv.getNum()-1+1;
    }
    // if vertix exist in hash map,
    // set corrospondenting newvi to the value in the hash map
    else
      newvi.set1Value(i, r-1);
  }
}

void ObjBody::convertIndex(SoMFInt32& fvi, const SoMFInt32& newvi) {
  for(int i=0; i<fvi.getNum(); i++)
    if(fvi[i]>=0) fvi.set1Value(i, newvi[fvi[i]]);
}

// complexibility: ?
bool ObjBody::TwoIndexHash::operator()(const TwoIndex& l1, const TwoIndex& l2) const {
  if(l1.vi1<l2.vi1) return true;
  else if(l1.vi1>l2.vi1) return false;
  else
    if(l1.vi2<l2.vi2) return true;
    else if(l1.vi2>l2.vi2) return false;
    else return false;
}

void ObjBody::computeNormals(const SoMFInt32& fvi, const SoMFVec3f &v, SoMFInt32& fni, SoMFVec3f& n, SoMFInt32& oli) {
  map<TwoIndex, vector<XXX>, TwoIndexHash> lni; // line normal index = index of normal of start/end point
  map<int, vector<int> > vni;

  for(int i=0; i<fvi.getNum(); i+=4) {
    // set face normals fn
    SbVec3f fn, ft, fb;
    ft=v[fvi[i+1]]-v[fvi[i+0]];
    fb=v[fvi[i+2]]-v[fvi[i+0]];
    fn=ft.cross(fb);
    fni.set1Value(i+0, i+0);
    fni.set1Value(i+1, i+1);
    fni.set1Value(i+2, i+2);
    fni.set1Value(i+3, -1);
    n.set1Value(i+0, fn);
    n.set1Value(i+1, fn);
    n.set1Value(i+2, fn);

    // store all face indexies of each line in a map
    TwoIndex l;
    XXX xxx;
    l.vi1=fvi[i+0]; l.vi2=fvi[i+1];
    xxx.ni1=i+0; xxx.ni2=i+1;
    if(l.vi1>l.vi2) { int dummy=l.vi1; l.vi1=l.vi2; l.vi2=dummy;
                          dummy=xxx.ni1; xxx.ni1=xxx.ni2; xxx.ni2=dummy; }
    lni[l].push_back(xxx);
    l.vi1=fvi[i+1]; l.vi2=fvi[i+2];
    xxx.ni1=i+1; xxx.ni2=i+2;
    if(l.vi1>l.vi2) { int dummy=l.vi1; l.vi1=l.vi2; l.vi2=dummy;
                          dummy=xxx.ni1; xxx.ni1=xxx.ni2; xxx.ni2=dummy; }
    lni[l].push_back(xxx);
    l.vi1=fvi[i+2]; l.vi2=fvi[i+0];
    xxx.ni1=i+2; xxx.ni2=i+0;
    if(l.vi1>l.vi2) { int dummy=l.vi1; l.vi1=l.vi2; l.vi2=dummy;
                          dummy=xxx.ni1; xxx.ni1=xxx.ni2; xxx.ni2=dummy; }
    lni[l].push_back(xxx);

    // vni
    vni[fvi[i+0]].push_back(i+0);
    vni[fvi[i+1]].push_back(i+1);
    vni[fvi[i+2]].push_back(i+2);
  }
  map<TwoIndex, vector<XXX>, TwoIndexHash>::iterator i;
  int ni1, ni2;
  SbVec3f nNew;
  for(i=lni.begin(); i!=lni.end(); i++) {
    if(i->second.size()!=2) continue;
    bool smooth=false;
    ni1=i->second[0].ni1; ni2=i->second[1].ni1;
    if(acos(n[fni[ni1]].dot(n[fni[ni2]])/n[fni[ni1]].length()/n[fni[ni2]].length())<smoothBarrier) {
      smooth=true;
      nNew=n[fni[ni1]]+n[fni[ni2]];
      n.set1Value(fni[ni1], nNew); n.set1Value(fni[ni2], nNew);
      vector<int> vvv=vni[i->first.vi1];
      for(int k=0; k<vvv.size(); k++)
        if(fni[vvv[k]]==fni[ni2]) fni.set1Value(vvv[k], fni[ni1]);
    }
    ni1=i->second[0].ni2; ni2=i->second[1].ni2;
    if(acos(n[fni[ni1]].dot(n[fni[ni2]])/n[fni[ni1]].length()/n[fni[ni2]].length())<smoothBarrier) {
      smooth=true;
      nNew=n[fni[ni1]]+n[fni[ni2]];
      n.set1Value(fni[ni1], nNew); n.set1Value(fni[ni2] ,nNew);
      vector<int> vvv=vni[i->first.vi1];
      for(int k=0; k<vvv.size(); k++)
        if(fni[vvv[k]]==fni[ni2]) fni.set1Value(vvv[k], fni[ni1]);
    }
    if(!smooth) {
      oli.set1Value(oli.getNum(), fvi[i->second[0].ni1]);
      oli.set1Value(oli.getNum(), fvi[i->second[0].ni2]);
      oli.set1Value(oli.getNum(), -1);
    }
  }
}
