#include <libxml/xmlschemas.h>
#include <libxml/xinclude.h>
#include <iostream>
#include <fstream>
extern "C" {
#include <regex.h>
}
#include "env.h"
#include "mbxmlutilstinyxml/tinyxml-src/tinyxml.h"
#include "mbxmlutilstinyxml/tinyxml-src/tinynamespace.h"
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>

#define MBXMLUTILSPVNS "{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}"

using namespace std;

char *nslocation;

int machinePrec;

// validate file using schema (currently by libxml)
int validate(const char *schema, const char *file) {
  xmlDoc *doc;
  cout<<"Parse and validate "<<file<<endl;
  doc=xmlParseFile(file);
  if(!doc) return 1;
  if(xmlXIncludeProcess(doc)<0) return 1;
  int ret=xmlSchemaValidateDoc(xmlSchemaNewValidCtxt(xmlSchemaParse(xmlSchemaNewParserCtxt(schema))), doc);
  if(ret!=0) return ret;
  xmlFreeDoc(doc);
  return 0;
}

int toOctave(TiXmlElement *&e) {
  if(e->ValueStr()==MBXMLUTILSPVNS"xmlMatrix") {
    string mat="[";
    for(TiXmlElement* row=e->FirstChildElement(); row!=0; row=row->NextSiblingElement()) {
      for(TiXmlElement* ele=row->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
        mat=mat+ele->GetText();
        if(ele->NextSiblingElement()) mat=mat+",";
      }
      if(row->NextSiblingElement()) mat=mat+";\n";
    }
    mat=mat+"]";
    TiXmlText *text=new TiXmlText(mat);
    e->Parent()->InsertEndChild(*text);
    e->Parent()->RemoveChild(e);
    e=0;
    return 0;
  }
  if(e->ValueStr()==MBXMLUTILSPVNS"xmlVector") {
    string vec="[";
    for(TiXmlElement* ele=e->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
      vec=vec+ele->GetText();
      if(ele->NextSiblingElement()) vec=vec+";";
    }
    vec=vec+"]";
    TiXmlText *text=new TiXmlText(vec);
    e->Parent()->InsertEndChild(*text);
    e->Parent()->RemoveChild(e);
    e=0;
    return 0;
  }
  if(e->ValueStr()==MBXMLUTILSPVNS"asciiVectorRef" || e->ValueStr()==MBXMLUTILSPVNS"asciiMatrixRef") {
    string filename=fixPath(e->GetElementWithXmlBase(0)->Attribute("xml:base"), e->Attribute("href"));
    ifstream file(filename.c_str());
    if(file.fail()) { cout<<e->GetElementWithXmlBase(0)->Attribute("xml:base")<<":"<<e->Row()<<": Can not open file: "<<filename<<endl; return 1; }
    string line, vec;
    while(!file.eof()) {;
      getline(file, line);
      // delete comments starting with % or # and append lines to vec separated by ;
      size_t pos=line.find_first_of("%#");
      if(pos!=string::npos) vec=vec+line.substr(0,pos)+";"; else vec=vec+line+";";
    }
    regex_t re;
    regmatch_t pmatch[1];
    // replace sequences of ; with ;
    regcomp(&re, ";( *;)+", REG_EXTENDED);
    while(regexec(&re, vec.c_str(), 1, pmatch, 0)==0)
      vec=vec.substr(0, pmatch[0].rm_so)+";"+vec.substr(pmatch[0].rm_eo);
    regfree(&re);
    // delete leading ;
    regcomp(&re, "^ *;", REG_EXTENDED);
    if(regexec(&re, vec.c_str(), 1, pmatch, 0)==0)
      vec=vec.substr(pmatch[0].rm_eo);
    regfree(&re);
    // delete tailing ;
    regcomp(&re, "; *$", REG_EXTENDED);
    if(regexec(&re, vec.c_str(), 1, pmatch, 0)==0)
      vec=vec.substr(0,pmatch[0].rm_so);
    regfree(&re);
    // convert ; to ;\n if we have a matrix
    if(e->ValueStr()==MBXMLUTILSPVNS"asciiMatrixRef") {
      size_t i;
      while((i=vec.find(';'))!=string::npos)
        vec=vec.substr(0,i)+"#\n"+vec.substr(i+1);
      while((i=vec.find('#'))!=string::npos)
        vec=vec.substr(0,i)+";"+vec.substr(i+1);
    }

    vec="["+vec+"]";
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

string octaveEval(string prestr, string str, bool exitOnError=true, bool clearOnStart=true) {
  int dummy;
  string clear="";
  if(clearOnStart) clear="clear all;\n";
  streambuf *orgcerr=std::cerr.rdbuf(0); // disable std::cerr
  eval_string(clear+prestr+"ret="+str,true,dummy,0);
  std::cerr.rdbuf(orgcerr); // enable std::cerr
  if(error_state!=0) {
    error_state=0;
    if(!exitOnError) std::cerr.rdbuf(0); // disable std::cerr if not exiting on error
    eval_string(clear+prestr+str,true,dummy,0);
    if(!exitOnError) std::cerr.rdbuf(orgcerr); // enable std::cerr if not exiting on error
    if(error_state!=0) {
      if(exitOnError)
        throw string("In octave expression: "+str);
      else {
        error_state=0;
        if(str.substr(0,6)=="error(")
          return str;
        else
          return string("error(\"")+str+"\")";
      }
    }
  }
  octave_value o=eval_string("ret;",true,dummy);
  if(error_state!=0)
    throw string("'ret' variable not set in octave statement list: "+str);
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
  else
    throw string("Unknown type in octave expression: "+str);

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
    size_t i;
    try {
      // fill octave with variables
      octaveEval("", "1;"); // clear all
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
    // check
    if((e->Attribute("href") && e->FirstChildElement()) ||
       (e->Attribute("href")==0 && e->FirstChildElement()==0)) {
      TiXml_location(e, "", ": Only the href attribute OR a child element is allowed in embed!");
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
    string countstr=string(e->Attribute("count"));
    countstr=octaveEval(paramString, countstr+";");
    int count=atoi(countstr.c_str());

    // couter name
    string counterName=e->Attribute("counterName");

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
      cout<<"Process xml[Matrix|Vector], ascii[Matrix|Vector]Ref elements in "<<file<<endl;
      if(toOctave(enew)!=0) {
        TiXml_location(e, "  included by: ", "");
        return 1;
      }
    }
    else // or take the child element (as a clone, because the embed element is deleted)
      enew=(TiXmlElement*)e->FirstChildElement()->Clone();

    // include a processing instruction with the line number of the original element
    TiXmlUnknown embedLine;
    embedLine.SetValue("?OriginalElementLineNr "+TiXml_itoa(e->Row())+"?");
    enew->InsertBeforeChild(enew->FirstChild(), embedLine);

    // delete embed element and insert count time the new element
    for(int i=1; i<=count; i++) {
      ostringstream istr; istr<<i;
      if(octaveEval(paramString+counterName+"="+istr.str()+";\n",onlyif)=="1") {
        cout<<"Embed "<<(file==""?"inline element":file)<<" ("<<i<<"/"<<count<<")"<<endl;
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
        cout<<"Skip embeding "<<(file==""?"inline element":file)<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
    }
    return 0;
  }
  else {
    if(e->GetText()) {
      // eval all text elements
      e->FirstChild()->SetValue(octaveEval(paramString, string(e->GetText())));
      // convert all text elements to SI unit
      if(e->Attribute("unit")) {
        e->FirstChild()->SetValue(octaveEval(string("value=")+e->GetText()+";\n", units[e->Attribute("unit")]));
        e->RemoveAttribute("unit");
      }
      e->FirstChild()->ToText()->SetCDATA(false);
    }
  
    // eval the "{...}" part in all name and ref* attributes
    TiXmlAttribute *a=e->FirstAttribute();
    for(TiXmlAttribute *a=e->FirstAttribute(); a!=0; a=a->Next()) {
      if(a->Name()==string("name") || string(a->Name()).substr(0,3)=="ref") {
        string s=a->ValueStr();
        int i;
        while((i=s.find('{'))>=0) {
          int j=s.find('}');
          s=s.substr(0,i)+octaveEval(paramString, s.substr(i+1,j-i-1))+s.substr(j+1);
        }
        a->SetValue(s);
      }
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
  int i1=dirfilename.find_last_of('/');
  int i2=dirfilename.find_last_of('\\');
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

  // initialize octave
  const char *octave_argv[2]={"dummy", "-q"};
  octave_main(2, (char**)octave_argv, 1);
  int dummy;
  streambuf *orgcerr=std::cerr.rdbuf(0); // disable std::cerr
  eval_string("warning(\"error\",\"Octave:divide-by-zero\");",true,dummy,0); // 1/0 is error
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
      if(validate(SCHEMADIR"/parameter.xsd", paramxml)!=0) return 1;

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
    cout<<"Process xml[Matrix|Vector], ascii[Matrix|Vector]Ref elements in "<<paramxml<<endl;
    if(string(paramxml)!="none")
      if(toOctave(paramxmlroot)!=0) return 1;

    // generate octave parameter string
    cout<<"Generate octave parameter string from "<<paramxml<<endl;
    string paramString="";
    if(string(paramxml)!="none")
      if(genParamString(paramxmlroot, paramString)!=0) return 1;

    // get units
    cout<<"Build unit list for measurements"<<endl;
    TiXmlDocument *mmdoc=new TiXmlDocument;
    mmdoc->LoadFile(XMLDIR"/measurement.xml");
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
    cout<<"Process xml[Matrix|Vector], ascii[Matrix|Vector]Ref elements in "<<mainxml<<endl;
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
