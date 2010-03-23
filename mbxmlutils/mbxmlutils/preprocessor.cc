#include <libxml/xmlschemas.h>
#include <libxml/xinclude.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include "env.h"
#include "mbxmlutilstinyxml/tinyxml-src/tinyxml.h"
#include "mbxmlutilstinyxml/tinyxml-src/tinynamespace.h"
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>

#define MBXMLUTILSPVNS "{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}"

using namespace std;

string SCHEMADIR;
string XMLDIR;
string OCTAVEDIR;

char *nslocation;

int machinePrec;

// validate file using schema (currently by libxml)
int validate(const string &schema, const char *file) {
  xmlDoc *doc;
  cout<<"Parse and validate "<<file<<endl;
  doc=xmlParseFile(file);
  if(!doc) return 1;
  if(xmlXIncludeProcess(doc)<0) return 1;
  int ret=xmlSchemaValidateDoc(xmlSchemaNewValidCtxt(xmlSchemaParse(xmlSchemaNewParserCtxt(schema.c_str()))), doc);
  if(ret!=0) return ret;
  xmlFreeDoc(doc);
  return 0;
}

int toOctave(TiXmlElement *&e) {
  if(e->ValueStr()==MBXMLUTILSPVNS"xmlMatrix") {
    string mat="[";
    for(TiXmlElement* row=e->FirstChildElement(); row!=0; row=row->NextSiblingElement()) {
      for(TiXmlElement* ele=row->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
        mat+=ele->GetText();
        if(ele->NextSiblingElement()) mat+=",";
      }
      if(row->NextSiblingElement()) mat+=";\n";
    }
    mat+="]";
    TiXmlText *text=new TiXmlText(mat);
    e->Parent()->InsertEndChild(*text);
    e->Parent()->RemoveChild(e);
    e=0;
    return 0;
  }
  if(e->ValueStr()==MBXMLUTILSPVNS"xmlVector") {
    string vec="[";
    for(TiXmlElement* ele=e->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
      vec+=ele->GetText();
      if(ele->NextSiblingElement()) vec+=";";
    }
    vec+="]";
    TiXmlText *text=new TiXmlText(vec);
    e->Parent()->InsertEndChild(*text);
    e->Parent()->RemoveChild(e);
    e=0;
    return 0;
  }

  TiXmlElement *c=e->FirstChildElement();
  while(c) {
    if(toOctave(c)!=0) return 1;
    if(c==0) break; // break if c was removed by toOctave at the line below
    c=c->NextSiblingElement();
  }

  return 0;
}

#define PATHLENGTH 10240
string octaveEval(string prestr, string str, bool exitOnError=true, bool clearOnStart=true, TiXmlElement *e=NULL) {
  static char savedPath[PATHLENGTH];
  if(e) { // set working dir to path of current file, so that octave works with correct relative paths
    if(getcwd(savedPath, PATHLENGTH)==0) throw(1);
    if(chdir(fixPath(e->GetElementWithXmlBase(0)->Attribute("xml:base"),".").c_str())!=0) throw(1);
  }

  int dummy;
  // delete leading new lines in str
  for(unsigned int i=0; i<str.length() && (str[i]==' ' || str[i]=='\n' || str[i]=='\t'); i++)
    str[i]=' ';

  string clear="";
  if(clearOnStart) clear="clear -all;\n";
  streambuf *orgcerr=std::cerr.rdbuf(0); // disable std::cerr
  eval_string(clear+prestr+"ret="+str,true,dummy,0);
  std::cerr.rdbuf(orgcerr); // enable std::cerr
  if(error_state!=0) {
    error_state=0;
    if(!exitOnError) std::cerr.rdbuf(0); // disable std::cerr if not exiting on error
    eval_string(clear+prestr+str,true,dummy,0);
    if(!exitOnError) std::cerr.rdbuf(orgcerr); // enable std::cerr if not exiting on error
    if(error_state!=0) {
      if(exitOnError) {
        if(e) if(chdir(savedPath)!=0) throw(1);
        throw string("In octave expression: "+str);
      }
      else {
        error_state=0;
        if(str.substr(0,6)=="error(") {
          if(e) if(chdir(savedPath)!=0) throw(1);
          return str;
        }
        else {
          if(e) if(chdir(savedPath)!=0) throw(1);
          return string("error(\"")+str+"\")";
        }
      }
    }
  }
  octave_value o=eval_string("ret;",true,dummy);
  if(error_state!=0) {
    if(e) if(chdir(savedPath)!=0) throw(1);
    throw string("'ret' variable not set in octave statement list: "+str);
  }
  ostringstream ret;
  ret.precision(machinePrec);
  if(o.is_scalar_type() && o.is_real_type())
    ret<<o.double_value();
  else if(o.is_matrix_type() && o.is_real_type()) {
    Matrix m=o.matrix_value();
    ret<<"[";
    for(int i=0; i<m.rows(); i++) {
      for(int j=0; j<m.cols(); j++)
        ret<<m(j*m.rows()+i)<<(j<m.cols()-1?",":"");
      ret<<(i<m.rows()-1?" ; ":"]");
    }
  }
  else if(o.is_string())
    ret<<"\""<<o.string_value()<<"\"";
  else {
    if(e) if(chdir(savedPath)!=0) throw(1);
    throw string("Unknown type in octave expression: "+str);
  }

  if(e) if(chdir(savedPath)!=0) throw(1);
  return ret.str().c_str();
}

struct Param {
  Param(string n, string eq, TiXmlElement *e) : name(n), equ(eq), ele(e) {}
  string name, equ;
  TiXmlElement *ele;
};
int genParamString(TiXmlElement *e, string &paramString) {
  // generate a vector of parameters
  vector<Param> param;
  for(TiXmlElement *ee=e->FirstChildElement(); ee!=0; ee=ee->NextSiblingElement())
    param.push_back(Param(ee->Attribute("name"), ee->GetText(), ee));

  // outer loop for resolving the tree structure of parameters
  for(size_t j=0; j<param.size(); j++) {
    size_t i=0;
    try {
      // fill octave with variables
      octaveEval("", "1;\n"+paramString); // clear all
      for(i=0; i<param.size(); i++)
        octaveEval("", param[i].name+"="+param[i].equ, false, false);
      // try to evaluate the parameter
      for(i=0; i<param.size(); i++)
        param[i].equ=octaveEval("", param[i].equ,(j==param.size()-1)?true:false, false);
    }
    catch(string str) {
      cout<<param[i].ele->GetElementWithXmlBase(0)<<":"<<param[i].ele->Row()<<": ";
      if(str.find("error(")!=string::npos)
        cout<<"Possible a infinite recursion in parameter file at e.g.: "<<param[i].ele->GetText()<<endl;
      else
        cout<<str<<endl;
      return 1;
    }
  }

  // generate the octave string
  for(size_t i=0; i<param.size(); i++)
    paramString=paramString+param[i].name+"="+param[i].equ+";\n";

  return 0;
}

int embed(TiXmlElement *&e, map<string,string> &nsprefix, string paramString, map<string,string> &units) {
try {
  if(e->ValueStr()==MBXMLUTILSPVNS"embed") {
    // check if only href OR child element (This is not checked by the schema)
    TiXmlElement *l=0, *dummy;
    for(dummy=e->FirstChildElement(); dummy!=0; l=dummy, dummy=dummy->NextSiblingElement());
    if((e->Attribute("href") && l && l->ValueStr()!=MBXMLUTILSPVNS"localParameter") ||
       (e->Attribute("href")==0 && (l==0 || l->ValueStr()==MBXMLUTILSPVNS"localParameter"))) {
      TiXml_location(e, "", ": Only the href attribute OR a child element (expect pv:localParameter) is allowed in embed!");
      return 1;
    }
    // check if attribute count AND counterName or none of both
    if((e->Attribute("count")==0 && e->Attribute("counterName")!=0) ||
       (e->Attribute("count")!=0 && e->Attribute("counterName")==0)) {
      TiXml_location(e, "", ": Only both, the count and counterName attribute must be given or none of both!");
      return 1;
    }

    // get file name if href attribute exist
    string file="";
    if(e->Attribute("href")) {
      file=fixPath(e->GetElementWithXmlBase(0)->Attribute("xml:base"), e->Attribute("href"));
    }

    // get onlyif attribute if exist
    string onlyif="1";
    if(e->Attribute("onlyif"))
      onlyif=e->Attribute("onlyif");

    // evaluate count using parameters
    int count=1;
    if(e->Attribute("count")) {
      string countstr=string(e->Attribute("count"));
      countstr=octaveEval(paramString, countstr+";", true, true, e);
      count=atoi(countstr.c_str());
    }

    // couter name
    string counterName="MBXMLUtilsDummyCounterName";
    if(e->Attribute("counterName"))
      counterName=e->Attribute("counterName");

    TiXmlElement *enew;
    // validate/load if file is given
    if(file!="") {
      if(validate(nslocation, file.c_str())!=0) {
        TiXml_location(e, "  included by: ", "");
        return 1;
      }
      cout<<"Read "<<file<<endl;
      TiXmlDocument *doc=new TiXmlDocument;
      doc->LoadFile(file.c_str());
      enew=doc->FirstChildElement();
      incorporateNamespace(enew, nsprefix);
      // convert embeded file to octave notation
      cout<<"Process xml[Matrix|Vector] elements in "<<file<<endl;
      if(toOctave(enew)!=0) {
        TiXml_location(e, "  included by: ", "");
        return 1;
      }
    }
    else { // or take the child element (as a clone, because the embed element is deleted)
      if(e->FirstChildElement()->ValueStr()==MBXMLUTILSPVNS"localParameter")
        enew=(TiXmlElement*)e->FirstChildElement()->NextSiblingElement()->Clone();
      else
        enew=(TiXmlElement*)e->FirstChildElement()->Clone();
      enew->SetAttribute("xml:base", e->GetElementWithXmlBase(0)->Attribute("xml:base")); // add a xml:base attribute
    }

    // include a processing instruction with the line number of the original element
    TiXmlUnknown embedLine;
    embedLine.SetValue("?OriginalElementLineNr "+TiXml_itoa(e->Row())+"?");
    enew->InsertBeforeChild(enew->FirstChild(), embedLine);


    // generate local paramter for embed
    if(e->FirstChildElement() && e->FirstChildElement()->ValueStr()==MBXMLUTILSPVNS"localParameter") {
      // check if only href OR p:parameter child element (This is not checked by the schema)
      if((e->FirstChildElement()->Attribute("href") && e->FirstChildElement()->FirstChildElement()) ||
         (!e->FirstChildElement()->Attribute("href") && !e->FirstChildElement()->FirstChildElement())) {
        TiXml_location(e->FirstChildElement(), "", ": Only the href attribute OR the child element p:parameter) is allowed here!");
        return 1;
      }
      cout<<"Generate local octave parameter string for "<<(file==""?"<inline element>":file)<<endl;
      if(e->FirstChildElement()->FirstChildElement()) // inline parameter
        genParamString(e->FirstChildElement()->FirstChildElement(), paramString);
      else { // parameter from href attribute
        string paramFile=e->FirstChildElement()->Attribute("href");
        // validate local parameter file
        if(validate(SCHEMADIR+"/http___openmbv_berlios_de_MBXMLUtils/parameter.xsd", paramFile.c_str())!=0) {
          TiXml_location(e->FirstChildElement(), "  included by: ", "");
          return 1;
        }
        // read local parameter file
        cout<<"Read local parameter file "<<paramFile<<endl;
        TiXmlDocument *localparamxmldoc=new TiXmlDocument;
        localparamxmldoc->LoadFile(paramFile.c_str());
        TiXmlElement *localparamxmlroot=localparamxmldoc->FirstChildElement();
        map<string,string> dummy;
        incorporateNamespace(localparamxmlroot,dummy);
        // generate local parameters
        genParamString(localparamxmlroot, paramString);
        delete localparamxmldoc;
      }
    }

    // delete embed element and insert count time the new element
    for(int i=1; i<=count; i++) {
      ostringstream istr; istr<<i;
      if(octaveEval(paramString+counterName+"="+istr.str()+";\n",onlyif)=="1") {
        cout<<"Embed "<<(file==""?"<inline element>":file)<<" ("<<i<<"/"<<count<<")"<<endl;
        if(i==1) e=(TiXmlElement*)(e->Parent()->ReplaceChild(e, *enew));
        else e=(TiXmlElement*)(e->Parent()->InsertAfterChild(e, *enew));

        // include a processing instruction with the count number
        TiXmlUnknown countNr;
        countNr.SetValue("?EmbedCountNr "+TiXml_itoa(i)+"?");
        e->InsertAfterChild(e->FirstChild(), countNr);
    
        // apply embed to new element
        if(embed(e, nsprefix, paramString+counterName+"="+istr.str()+";\n",units)!=0) return 1;
      }
      else
        cout<<"Skip embeding "<<(file==""?"<inline element>":file)<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
    }
    return 0;
  }
  else {
    // THIS IS A WORKAROUND! Actually not all Text-Elements should be converted but only the Text-Elements
    // of XML elementx of a type devived from pv:scalar, pv:vector, pv:matrix and pv:string. But for that a
    // schema aware processor is needed!
    if(e->GetText()) {
      // eval all text elements
      e->FirstChild()->SetValue(octaveEval(paramString, string(e->GetText()), true, true, e));
      // convert all text elements to SI unit
      if(e->Attribute("unit")) {
        e->FirstChild()->SetValue(octaveEval(string("value=")+e->GetText()+";\n", units[e->Attribute("unit")], true, true, e));
        e->RemoveAttribute("unit");
      }
      // evaluate convertUnit attribute if given
      if(e->Attribute("convertUnit")) {
        e->FirstChild()->SetValue(octaveEval(string("value=")+e->GetText()+";\n", e->Attribute("convertUnit"), true, true, e));
        e->RemoveAttribute("convertUnit");
      }
      e->FirstChild()->ToText()->SetCDATA(false);
    }
  
    // THIS IS A WORKAROUND! Actually not all 'name' and 'ref*' attributes should be converted but only the
    // XML attributes of a type devived from pv:fullOctaveString and pv:partialOctaveString. But for that a
    // schema aware processor is needed!
    for(TiXmlAttribute *a=e->FirstAttribute(); a!=0; a=a->Next())
      if(a->Name()==string("name") || string(a->Name()).substr(0,3)=="ref") {
        string s=a->ValueStr();
        int i;
        while((i=s.find('{'))>=0) {
          int j=s.find('}');
          s=s.substr(0,i)+octaveEval(paramString, s.substr(i+1,j-i-1), true, true, e)+s.substr(j+1);
        }
        a->SetValue(s);
      }
  }

  TiXmlElement *c=e->FirstChildElement();
  while(c) {
    if(embed(c, nsprefix, paramString, units)!=0) return 1;
    c=c->NextSiblingElement();
  }
}
catch(string str) {
  TiXml_location(e, "", ": "+str);
  return 1;
}
 
  return 0;
}

string extractFileName(string dirfilename) {
  size_t i1=dirfilename.find_last_of('/');
  size_t i2=dirfilename.find_last_of('\\');
  i1=(i1==string::npos?-1:i1);
  i2=(i2==string::npos?-1:i2);
  i1=max(i1,i2);
  return dirfilename.substr(i1+1);
}

int main(int argc, char *argv[]) {
  if((argc-1)%3!=0 || argc<=1) {
    cout<<"Usage:"<<endl
        <<"mbxmlutilspp <param-file> [dir/]<main-file> <namespace-location-of-main-file>"<<endl
        <<"             [<param-file> [dir/]<main-file> <namespace-location-of-main-file>] ..."<<endl
        <<"  The output file is named '.pp.<main-file>'."<<endl
        <<"  Use 'none' if not <param-file> is avaliabel."<<endl
        <<""<<endl
        <<"Copyright (C) 2009 Markus Friedrich <mafriedrich@users.berlios.de>"<<endl
        <<"This is free software; see the source for copying conditions. There is NO"<<endl
        <<"warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl
        <<""<<endl
        <<"Licensed under the GNU Lesser General Public License (LGPL)"<<endl;
    return 0;
  }

  // check for environment variables (none default installation)
  char *env;
  SCHEMADIR=SCHEMADIR_DEFAULT;
  if((env=getenv("MBXMLUTILSSCHEMADIR"))) SCHEMADIR=env;
  XMLDIR=XMLDIR_DEFAULT;
  if((env=getenv("MBXMLUTILSXMLDIR"))) XMLDIR=env;
  OCTAVEDIR=OCTAVEDIR_DEFAULT;
  if((env=getenv("MBXMLUTILSOCTAVEDIR"))) OCTAVEDIR=env;

  // initialize octave
  const char *octave_argv[2]={"dummy", "-q"};
  octave_main(2, (char**)octave_argv, 1);
  int dummy;
  streambuf *orgcerr=std::cerr.rdbuf(0); // disable std::cerr
  eval_string("warning(\"error\",\"Octave:divide-by-zero\");",true,dummy,0); // 1/0 is error
  error_state=0;
  eval_string("addpath(\""+OCTAVEDIR+"\");",true,dummy,0); // for octave >= 3.0.0
  error_state=0;
  eval_string("LOADPATH=[LOADPATH \":"+OCTAVEDIR+"\"];",true,dummy,0); // for octave < 3.0.0
  error_state=0;
  std::cerr.rdbuf(orgcerr); // enable std::cerr

  // preserve whitespace and newline in TiXmlText nodes
  TiXmlBase::SetCondenseWhiteSpace(false);

  // calcaulate machine precision
  double machineEps;
  for(machineEps=1.0; (1.0+machineEps)>1.0; machineEps*=0.5);
  machineEps*=2.0;
  machinePrec=(int)(-log(machineEps)/log(10))+1;

  // loop over all files
  for(int nr=0; nr<(argc-1)/3; nr++) {
    char *paramxml=argv[3*nr+1];
    char *mainxml=argv[3*nr+2];
    nslocation=argv[3*nr+3];

    // validate parameter file
    if(string(paramxml)!="none")
      if(validate(SCHEMADIR+"/http___openmbv_berlios_de_MBXMLUtils/parameter.xsd", paramxml)!=0) return 1;

    // read parameter file
    TiXmlElement *paramxmlroot=0;
    if(string(paramxml)!="none") {
      cout<<"Read "<<paramxml<<endl;
      TiXmlDocument *paramxmldoc=new TiXmlDocument;
      paramxmldoc->LoadFile(paramxml);
      paramxmlroot=paramxmldoc->FirstChildElement();
      map<string,string> dummy;
      incorporateNamespace(paramxmlroot,dummy);
    }

    // convert parameter file to octave notation
    if(string(paramxml)!="none") {
      cout<<"Process xml[Matrix|Vector] elements in "<<paramxml<<endl;
      if(toOctave(paramxmlroot)!=0) return 1;
    }

    // generate octave parameter string
    string paramString="";
    if(string(paramxml)!="none") {
      cout<<"Generate octave parameter string from "<<paramxml<<endl;
      if(genParamString(paramxmlroot, paramString)!=0) return 1;
    }

    // THIS IS A WORKAROUND! See before.
    // get units
    cout<<"Build unit list for measurements"<<endl;
    TiXmlDocument *mmdoc=new TiXmlDocument;
    mmdoc->LoadFile(XMLDIR+"/measurement.xml");
    TiXmlElement *ele, *el2;
    map<string,string> units;
    for(ele=mmdoc->FirstChildElement()->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement())
      for(el2=ele->FirstChildElement(); el2!=0; el2=el2->NextSiblingElement()) {
        if(units.find(el2->Attribute("name"))!=units.end()) {
          cout<<"ERROR! Unit name "<<el2->Attribute("name")<<" is defined more than once."<<endl;
          return 1;
        }
        units[el2->Attribute("name")]=el2->GetText();
      }

    // validate main file
    if(validate(nslocation, mainxml)!=0) return 1;

    // read main file
    cout<<"Read "<<mainxml<<endl;
    TiXmlDocument *mainxmldoc=new TiXmlDocument;
    mainxmldoc->LoadFile(mainxml);
    TiXmlElement *mainxmlroot=mainxmldoc->FirstChildElement();
    map<string,string> nsprefix;
    incorporateNamespace(mainxmlroot,nsprefix);

    // convert main file to octave notation
    cout<<"Process xml[Matrix|Vector] elements in "<<mainxml<<endl;
    if(toOctave(mainxmlroot)!=0) return 1;

    // embed/validate/toOctave/unit/eval files
    if(embed(mainxmlroot,nsprefix,paramString,units)!=0) return 1;

    // save result file
    cout<<"Save preprocessed file "<<mainxml<<" as .pp."<<extractFileName(mainxml)<<endl;
    TiXml_addLineNrAsProcessingInstruction(mainxmlroot);
    unIncorporateNamespace(mainxmlroot, nsprefix);
    mainxmldoc->SaveFile(".pp."+string(extractFileName(mainxml)));

    // validate preprocessed file
    if(validate(nslocation, (".pp."+string(extractFileName(mainxml))).c_str())!=0) return 1;
  }

  return 0;
}
