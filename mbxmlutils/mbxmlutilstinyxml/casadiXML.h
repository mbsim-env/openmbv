#include "mbxmlutilstinyxml/tinyxml.h"
#include <casadi/symbolic/sx/sx.hpp>
#include <casadi/symbolic/fx/sx_function.hpp>
#include <set>

#define MBXMLUTILSCASADINS_ "http://openmbv.berlios.de/MBXMLUtils/CasADi"
#define MBXMLUTILSCASADINS "{"MBXMLUTILSCASADINS_"}"

namespace CasADi {

inline MBXMLUtils::TiXmlElement* convertCasADiToXML(const CasADi::SX &s, std::map<CasADi::SXNode*, int> &nodes) {
  // add the node of s to the list of all nodes (creates a integer id for newly added nodes)
  std::string idStr;
  std::pair<std::map<CasADi::SXNode*, int>::iterator, bool> ret=nodes.insert(std::make_pair(s.get(), nodes.size()));
  // if the node of s already exists in the list of all nodes write a reference to this node to XML
  if(ret.second==false) {
    MBXMLUtils::TiXmlElement *e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"reference");
    std::stringstream str;
    str<<ret.first->second;
    e->SetAttribute("refid", str.str());
    return e;
  }
  // if the node of s does not exist in the list of all nodes set the id of this node
  else {
    std::stringstream str;
    str<<ret.first->second;
    idStr=str.str();
  }

  // add s to XML dependent on the type of s
  MBXMLUtils::TiXmlElement *e;
  if(s.isSymbolic()) {
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"SymbolicSX");
    e->LinkEndChild(new MBXMLUtils::TiXmlText(s.getName()));
  }
  else if(s.isZero())
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"ZeroSX");
  else if(s.isOne())
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"OneSX");
  else if(s.isMinusOne())
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"MinusOneSX");
  else if(s.isInf())
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"InfSX");
  else if(s.isMinusInf())
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"MinusInfSX");
  else if(s.isNan())
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"NanSX");
  else if(s.isInteger()) {
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"IntegerSX");
    std::stringstream str;
    str<<s.getIntValue();
    e->LinkEndChild(new MBXMLUtils::TiXmlText(str.str()));
  }
  else if(s.isConstant()) {
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"RealtypeSX");
    std::stringstream str;
    str.precision(18);
    str<<s.getValue();
    e->LinkEndChild(new MBXMLUtils::TiXmlText(str.str()));
  }
  else if(s.hasDep() && s.getNdeps()==2) {
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"BinarySX");
    e->SetAttribute("op", s.getOp());
    e->LinkEndChild(convertCasADiToXML(s.getDep(0), nodes));
    e->LinkEndChild(convertCasADiToXML(s.getDep(1), nodes));
  }
  else if(s.hasDep() && s.getNdeps()==1) {
    e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"UnarySX");
    e->SetAttribute("op", s.getOp());
    e->LinkEndChild(convertCasADiToXML(s.getDep(0), nodes));
  }
  else
    throw std::runtime_error("Unknown CasADi::SX type in convertCasADiToXML");

  // write also the id of a newly node to XML
  e->SetAttribute("id", idStr);
  return e;
}

inline MBXMLUtils::TiXmlElement* convertCasADiToXML(const CasADi::SXMatrix &m, std::map<CasADi::SXNode*, int> &nodes) {
  // if it is a scalar print it as a scalar
  if(m.size1()==1 && m.size2()==1)
    return convertCasADiToXML(m.elem(0, 0), nodes);
  // write each matrixRow of m to XML enclosed by a <matrixRow> element and each element in such rows 
  // to this element
  MBXMLUtils::TiXmlElement *e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"SXMatrix");
  if(m.size1()==1) e->SetAttribute("rowVector", "true");
  if(m.size2()==1) e->SetAttribute("columnVector", "true");
  for(int r=0; r<m.size1(); r++) {
    MBXMLUtils::TiXmlElement *matrixRow;
    if(m.size1()==1 || m.size2()==1)
      matrixRow=e;
    else {
      matrixRow=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"matrixRow");
      e->LinkEndChild(matrixRow);
    }
    for(int c=0; c<m.size2(); c++)
      matrixRow->LinkEndChild(convertCasADiToXML(m.elem(r, c), nodes));
  }
  return e;
}

inline MBXMLUtils::TiXmlElement* convertCasADiToXML(const CasADi::SXFunction &f) {
  // write each input of f to XML enclosed by a <inputs> element
  std::map<CasADi::SXNode*, int> nodes;
  MBXMLUtils::TiXmlElement *e=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"SXFunction");
  const std::vector<CasADi::SXMatrix> &in=f.inputExpr();
  MBXMLUtils::TiXmlElement *input=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"inputs");
  e->LinkEndChild(input);
  for(size_t i=0; i<in.size(); i++)
    input->LinkEndChild(convertCasADiToXML(in[i], nodes));
  // write each output of f to XML enclosed by a <outputs> element
  const std::vector<CasADi::SXMatrix> &out=f.outputExpr();
  MBXMLUtils::TiXmlElement *output=new MBXMLUtils::TiXmlElement(MBXMLUTILSCASADINS"outputs");
  e->LinkEndChild(output);
  for(size_t i=0; i<out.size(); i++)
    output->LinkEndChild(convertCasADiToXML(out[i], nodes));

  return e;
}

inline CasADi::SX createCasADiSXFromXML(MBXMLUtils::TiXmlElement *e, std::map<int, CasADi::SXNode*> &nodes) {
  // creata an SX dependent on the type
  CasADi::SX sx;
  if(e->ValueStr()==MBXMLUTILSCASADINS"BinarySX") {
    int op;
    e->QueryIntAttribute("op", &op);
    MBXMLUtils::TiXmlElement *ee=e->FirstChildElement();
    CasADi::SX dep0=createCasADiSXFromXML(ee, nodes);
    ee=ee->NextSiblingElement();
    CasADi::SX dep1=createCasADiSXFromXML(ee, nodes);
    sx=CasADi::SX::binary(op, dep0, dep1);
  }
  else if(e->ValueStr()==MBXMLUTILSCASADINS"UnarySX") {
    int op;
    e->QueryIntAttribute("op", &op);
    MBXMLUtils::TiXmlElement *ee=e->FirstChildElement();
    CasADi::SX dep=createCasADiSXFromXML(ee, nodes);
    sx=CasADi::SX::unary(op, dep);
  }
  else if(e->ValueStr()==MBXMLUTILSCASADINS"SymbolicSX") {
    MBXMLUtils::TiXmlText *ee=e->FirstChild()->ToText();
    sx=CasADi::SX(ee->ValueStr());
  }
  else if(e->ValueStr()==MBXMLUTILSCASADINS"RealtypeSX") {
    MBXMLUtils::TiXmlText *ee=e->FirstChild()->ToText();
    std::stringstream str(ee->ValueStr());
    double value;
    str>>value;
    sx=value;
  }
  else if(e->ValueStr()==MBXMLUTILSCASADINS"IntegerSX") {
    MBXMLUtils::TiXmlText *ee=e->FirstChild()->ToText();
    std::stringstream str(ee->ValueStr());
    int value;
    str>>value;
    sx=value;
  }
  else if(e->ValueStr()==MBXMLUTILSCASADINS"ZeroSX")
    sx=CasADi::casadi_limits<CasADi::SX>::zero;
  else if(e->ValueStr()==MBXMLUTILSCASADINS"OneSX")
    sx=CasADi::casadi_limits<CasADi::SX>::one;
  else if(e->ValueStr()==MBXMLUTILSCASADINS"MinusOneSX")
    sx=CasADi::casadi_limits<CasADi::SX>::minus_one;
  else if(e->ValueStr()==MBXMLUTILSCASADINS"InfSX")
    sx=CasADi::casadi_limits<CasADi::SX>::inf;
  else if(e->ValueStr()==MBXMLUTILSCASADINS"MinusInfSX")
    sx=CasADi::casadi_limits<CasADi::SX>::minus_inf;
  else if(e->ValueStr()==MBXMLUTILSCASADINS"NanSX")
    sx=CasADi::casadi_limits<CasADi::SX>::nan;
  // reference elements must be handled specially: return the referenced node instead of creating a new one
  else if(e->ValueStr()==MBXMLUTILSCASADINS"reference") {
    int refid;
    e->QueryIntAttribute("refid", &refid);
    sx=CasADi::SX(nodes[refid], false);
    return sx;
  }
  else
    throw std::runtime_error("Unknown XML element named "+e->ValueStr()+" in createCasADiSXFromXML");

  // insert a newly created SX (node) to the list of all nodes
  int id;
  e->QueryIntAttribute("id", &id);
  nodes.insert(std::make_pair(id, sx.get()));
  return sx;
}

inline CasADi::SXMatrix createCasADiSXMatrixFromXML(MBXMLUtils::TiXmlElement *e, std::map<int, CasADi::SXNode*> &nodes) {
  // create a SXMatrix
  if(e->ValueStr()==MBXMLUTILSCASADINS"SXMatrix") {
    // loop over all rows
    std::vector<std::vector<CasADi::SX> > ret;
    MBXMLUtils::TiXmlElement *matrixRow=e->FirstChildElement();
    while(matrixRow) {
      // loop over all elements in a matrixRow
      std::vector<CasADi::SX> stdrow;
      MBXMLUtils::TiXmlElement *matrixEle;
      if((e->Attribute("rowVector") && e->Attribute("rowVector")==std::string("true")) ||
         (e->Attribute("columnVector") && e->Attribute("columnVector")==std::string("true")))
        matrixEle=matrixRow;
      else
        matrixEle=matrixRow->FirstChildElement();
      while(matrixEle) {
        stdrow.push_back(createCasADiSXFromXML(matrixEle, nodes));
        matrixEle=matrixEle->NextSiblingElement();
      }
      if((e->Attribute("rowVector") && e->Attribute("rowVector")==std::string("true")) ||
         (e->Attribute("columnVector") && e->Attribute("columnVector")==std::string("true")))
        matrixRow=matrixEle;
      else
        matrixRow=matrixRow->NextSiblingElement();
      ret.push_back(stdrow);
    }
    if(e->Attribute("columnVector") && e->Attribute("columnVector")==std::string("true"))
      return CasADi::SXMatrix(ret).trans();
    else
      return CasADi::SXMatrix(ret);
  }
  // if it is a scalar create a SX
  else
    return createCasADiSXFromXML(e, nodes);
}

inline CasADi::SXFunction createCasADiSXFunctionFromXML(MBXMLUtils::TiXmlElement *e) {
  // create a SXFunction
  std::map<int, CasADi::SXNode*> nodes;
  if(e->ValueStr()==MBXMLUTILSCASADINS"SXFunction") {
    // get all inputs
    std::vector<CasADi::SXMatrix> in;
    MBXMLUtils::TiXmlElement *input=e->FirstChildElement();
    MBXMLUtils::TiXmlElement *inputEle=input->FirstChildElement();
    while(inputEle) {
      in.push_back(createCasADiSXMatrixFromXML(inputEle, nodes));
      inputEle=inputEle->NextSiblingElement();
    }
    // get all outputs
    std::vector<CasADi::SXMatrix> out;
    MBXMLUtils::TiXmlElement *output=input->NextSiblingElement();
    MBXMLUtils::TiXmlElement *outputEle=output->FirstChildElement();
    while(outputEle) {
      out.push_back(createCasADiSXMatrixFromXML(outputEle, nodes));
      outputEle=outputEle->NextSiblingElement();
    }

    return CasADi::SXFunction(in, out);
  }
  else
    throw std::runtime_error("Unknown XML element named "+e->ValueStr()+" in createCasADiSXFunctionFromXML.");
}

}
