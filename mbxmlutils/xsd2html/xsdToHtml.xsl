<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema"
  xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"
  xmlns:html="http://www.w3.org/1999/xhtml"
  xmlns="http://www.w3.org/1999/xhtml"
  version="1.0">

  <!-- If changes in this file are made, then the analog changes must
       be done in the file xstToTex.xsl -->

  <xsl:param name="PROJECT"/>
  <xsl:param name="PHYSICALVARIABLEHTMLDOC"/>
  <xsl:param name="INCLUDEDOXYGEN"/>



  <!-- output method -->
  <xsl:output method="xml"
    encoding="UTF-8"
    doctype-public="-//W3C//DTD XHTML 1.0 Transitional//EN"
    doctype-system="http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <xsl:template match="/">
    <!-- html header -->
    <html xml:lang="en" lang="en">
    <head>
      <title><xsl:value-of select="$PROJECT"/> - XML Documentation</title>
      <style type="text/css">
        div.para { margin-bottom:1ex }
        dl,dd { }
        dt { font-weight:bold }
        div.eqn { margin-bottom:1ex }
        img.eqn { }
        img.inlineeqn { }
        div.htmlfigure,table,caption,tr,td { }
        img.htmlfigure { }
        object.latexfigure { }

        h2,h3 { margin-top:10ex;margin;font-size:14pt }
        ul.content { padding-left:3ex;list-style-type:none }
        span.occurance { font-style:italic }

        *.element { font-family:monospace;font-weight:bold }
        *.type { font-family:monospace }
        *.attribute { font-family:monospace;font-weight:bold;margin-left:2ex }
        *.elementdocu { padding-left:3ex;margin:0;margin-bottom:1ex }
        *.elementdocu { margin-left:3ex;margin-bottom:1ex }
        *.classdocu { margin-bottom:1ex }
        ul.elementchoice { list-style-type:none;border-left-style:solid;border-left-color:blue;padding:0.1ex;margin-top:0.25ex;margin-bottom:0.25ex }
        *.elementchoicecolor { color:blue }
        ul.elementsequence { list-style-type:none;border-left-style:solid;border-left-color:red;padding:0.1ex;margin-top:0.25ex;margin-bottom:0.25ex }
        *.elementsequencecolor { color:red }
        ul.elementchild { list-style-type:none;padding-left:4ex }
        ul.elementsofclass { list-style-type:none;padding:0 }
      </style>
    </head>
    <body>
    <h1><xsl:value-of select="$PROJECT"/> - XML Documentation</h1>
    <h2>Contents</h2>
    <ul class="content">
      <li>1 <a name="content-introduction" href="#introduction">Introduction</a></li>
      <li>2 <a name="content-nomenclature" href="#nomenclature">Nomenclature</a></li>
      <li>3 Elements
        <ul class="content">
          <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[not(@substitutionGroup)]|/xs:schema/xs:element[not(@substitutionGroup=/xs:schema/xs:element/@name)]">
            <xsl:with-param name="LEVEL" select="0"/>
            <xsl:with-param name="LEVELNR" select="'3'"/>
            <xsl:sort select="@name"/>
          </xsl:apply-templates>
        </ul>
      </li>
    </ul>
    <h2>1 <a name="introduction" href="#content-introduction">Introduction:</a></h2>
    <xsl:apply-templates mode="CLASSANNOTATION" select="/xs:schema/xs:annotation/xs:documentation"/>
    <h2>2 <a name="nomenclature" href="#content-nomenclature">Nomenclature:</a></h2>
    <h3>2.1 A element:</h3>
    <p><span class="element">&lt;ElementName&gt;</span> <span class="occurance">[0-2]</span> (Type: <span class="type">elementType</span>)
    <br/><span class="attribute">attrName1</span> <span class="occurance">[required]</span> (Type: <span class="type">typeOfTheAttribute</span>)
    <br/><span class="attribute">attrName2</span> <span class="occurance">[optional]</span> (Type: <span class="type">typeOfTheAttribute</span>)</p>
    <p class="elementdocu">
      Documentation of the element.
    </p>
    <p>The upper nomenclature defines a XML element named <span class="element">ElementName</span> with (if given) a minimal occurance of 0 and a maximal occurance of 2. The element is of type <span class="type">elementType</span>.<br/>
    A occurance of <span class="occurance">[optional]</span> means <span class="occurance">[0-1]</span>.<br/>
    The element has two attributes named <span class="attribute">attrName1</span> and <span class="attribute">attrName2</span> of type <span class="type">typeOfTheAttribute</span>. A attribute can be optional or required.</p>
    <h3>2.2 A choice of element:</h3>
    <ul class="elementchoice">
      <li class="elementchoicecolor"><span class="occurance">[1-2]</span></li>
      <li><span class="element">&lt;ElemenetA&gt;</span></li>
      <li><span class="element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a choice of elements. Only one element of the given ones can be used. The choice has, if given, a minimal occurance of 1 and a maximal maximal occurence of 2.<br/>
    A occurance of <span class="occurance">[optional]</span> means <span class="occurance">[0-1]</span>.</p>
    <h3>2.3 A seqence of elements:</h3>
    <ul class="elementsequence">
      <li class="elementsequencecolor"><span class="occurance">[0-3]</span></li>
      <li><span class="element">&lt;ElemenetA&gt;</span></li>
      <li><span class="element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a sequence of elements. Each element must be given in that order. The sequence has, if given, a minimal occurance of 0 and a maximal maximal occurence of 3.<br/>
    A occurance of <span class="occurance">[optional]</span> means <span class="occurance">[0-1]</span>.</p>
    <h3>2.4 Nested sequences/choices:</h3>
    <ul class="elementsequence">
      <li class="elementsequencecolor"><span class="occurance">[1-2]</span></li>
      <li><span class="element">&lt;ElemenetA&gt;</span></li>
      <li>
        <ul class="elementchoice">
          <li class="elementchoicecolor"><span class="occurance">[0-3]</span></li>
          <li><span class="element">&lt;ElemenetC&gt;</span></li>
          <li><span class="element">&lt;ElemenetD&gt;</span></li>
        </ul>
      </li>
      <li><span class="element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>Sequences and choices can be nested like above.</p>
    <h3>2.5 Child Elements:</h3>
    <ul class="elementsequence">
      <li class="elementsequencecolor"><span class="occurance">[1-2]</span></li>
      <li><span class="element">&lt;ParantElemenet&gt;</span>
        <ul class="elementchild">
          <li>
            <ul class="elementchoice">
              <li class="elementchoicecolor"><span class="occurance">[0-3]</span></li>
              <li><span class="element">&lt;ChildElemenetA&gt;</span></li>
              <li><span class="element">&lt;ChildElemenetB&gt;</span></li>
            </ul>
          </li>
        </ul>
      </li>
    </ul>
    <p>A indent indicates child elements for a given element.</p>

    <h2>3 Elements</h2>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[not(@substitutionGroup)]|/xs:schema/xs:element[not(@substitutionGroup=/xs:schema/xs:element/@name)]">
      <xsl:with-param name="LEVEL" select="0"/>
      <xsl:with-param name="LEVELNR" select="'3'"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>
    </body></html>
  </xsl:template>

  <!-- generate contents -->
  <xsl:template mode="CONTENT" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="LEVELNR"/>
    <xsl:param name="NAME" select="@name"/>
    <li>
      <xsl:value-of select="$LEVELNR"/>.<xsl:value-of select="position()"/>
      <xsl:text> </xsl:text>
      <a class="element"><xsl:attribute name="name">content-<xsl:value-of select="@name"/></xsl:attribute>
        <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute>&lt;<xsl:value-of select="@name"/>&gt;</a>
      <xsl:if test="/xs:schema/xs:element[@substitutionGroup=$NAME]">
        <ul class="content">
          <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[@substitutionGroup=$NAME]">
            <xsl:with-param name="LEVEL" select="$LEVEL+1"/>
            <xsl:with-param name="LEVELNR" select="concat($LEVELNR,'.',position())"/>
            <xsl:sort select="@name"/>
          </xsl:apply-templates>
        </ul>
      </xsl:if>
    </li>
  </xsl:template>

  <!-- walk throw all elements -->
  <xsl:template mode="WALKCLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="LEVELNR"/>
    <xsl:param name="NAME" select="@name"/>
    <xsl:apply-templates mode="CLASS" select=".">
      <xsl:with-param name="LEVEL" select="$LEVEL"/>
      <xsl:with-param name="TITLENR" select="concat($LEVELNR,'.',position())"/>
    </xsl:apply-templates>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[@substitutionGroup=$NAME]">
      <xsl:with-param name="LEVEL" select="$LEVEL+1"/>
      <xsl:with-param name="LEVELNR" select="concat($LEVELNR,'.',position())"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>
  </xsl:template>

  <!-- class -->
  <xsl:template mode="CLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="TITLENR"/>
    <xsl:param name="TYPENAME" select="@type"/>
    <xsl:param name="CLASSNAME" select="@name"/>
    <!-- heading -->
    <xsl:element name="h3">
      <xsl:attribute name="class">element</xsl:attribute>
      <xsl:value-of select="$TITLENR"/>
      <xsl:text> </xsl:text>
      <a>
        <xsl:attribute name="name">
          <xsl:value-of select="@name"/>
        </xsl:attribute>
        <xsl:attribute name="href">#content-<xsl:value-of select="@name"/></xsl:attribute>
        &lt;<xsl:value-of select="@name"/>&gt;
      </a>
    </xsl:element>
    <table border="1">
      <!-- abstract -->
      <tr><td>Abstract Element:</td><td>
        <xsl:if test="@abstract='true'">true</xsl:if>
        <xsl:if test="@abstract!='true' or not(@abstract)">false</xsl:if>
      </td></tr>
      <!-- inherits -->
      <tr><td>Inherits:</td><td>
        <xsl:if test="@substitutionGroup">
          <a class="element">
            <xsl:attribute name="href">#<xsl:value-of select="@substitutionGroup"/></xsl:attribute>
            &lt;<xsl:value-of select="@substitutionGroup"/>&gt;</a>
        </xsl:if>
      </td></tr>
      <!-- inherited by -->
      <tr><td>Inherited by:</td><td>
        <xsl:if test="count(/xs:schema/xs:element[@substitutionGroup=$CLASSNAME])>0">
          <xsl:for-each select="/xs:schema/xs:element[@substitutionGroup=$CLASSNAME]">
            <xsl:sort select="@name"/>
            <a class="element">
              <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute>
              &lt;<xsl:value-of select="@name"/>&gt;</a>, 
          </xsl:for-each>
        </xsl:if>
      </td></tr>
      <!-- used in -->
      <!--<tr><td>Can be used in:</td><td><xsl:apply-templates mode="USEDIN2" select="."/></td></tr>-->
      <!-- class attributes -->
      <tr><td>Attributes:</td><td>
      <xsl:apply-templates mode="CLASSATTRIBUTE" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:attribute"/>
      </td></tr>
    </table>
    <!-- class documentation -->
    <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
    <!-- child elements -->
    <p>Child Elements:</p>
    <!-- child elements for not base class -->
    <xsl:apply-templates mode="CLASS" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension">
      <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
    </xsl:apply-templates>
    <!-- child elements for base class -->
    <xsl:if test="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
      <ul class="elementsofclass">
        <xsl:apply-templates mode="SIMPLECONTENT" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </xsl:if>
  </xsl:template>

  <!-- class attributes -->
  <xsl:template mode="CLASSATTRIBUTE" match="/xs:schema/xs:complexType/xs:attribute">
    <span class="element"><xsl:value-of select="@name"/></span>
    <xsl:if test="@use='required'">
      <span class="occurance"> [required]</span><xsl:text> </xsl:text>
    </xsl:if>
    <xsl:if test="@use!='required'">
      <span class="occurance"> [optional]</span><xsl:text> </xsl:text>
    </xsl:if>
    (Type: <a class="type">
      <xsl:attribute name="href">
        #<xsl:value-of select="@type"/>
      </xsl:attribute>
      <xsl:value-of select="@type"/>
    </a>)
    <br/>
  </xsl:template>

  <!-- used in -->
  <!--<xsl:template mode="USEDIN2" match="/xs:schema/xs:element">
      <xsl:param name="SUBSTGROUP" select="@substitutionGroup"/>
      <xsl:param name="CLASSNAME" select="@name"/>
      <xsl:apply-templates mode="USEDIN" select="/descendant::xs:element[@ref=$CLASSNAME]"/>
      <xsl:apply-templates mode="USEDIN2" select="/xs:schema/xs:element[@name=$SUBSTGROUP]"/>
  </xsl:template>
  <xsl:template mode="USEDIN" match="xs:element">
    <xsl:apply-templates mode="USEDIN" select="ancestor::xs:complexType[last()]"/>
  </xsl:template>
  <xsl:template mode="USEDIN" match="xs:complexType">
    <xsl:param name="CLASSTYPE" select="@name"/>
    <a class="element">
      <xsl:attribute name="href">#<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/></xsl:attribute>
      &lt;<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>&gt;</a>,
  </xsl:template>-->

  <!-- child elements for not base class -->
  <xsl:template mode="CLASS" match="xs:extension">
    <xsl:param name="CLASSNAME"/>
    <ul class="elementsofclass">
      <!-- elements from base class -->
      <li>
        All Elements from 
        <a class="element">
          <xsl:attribute name="href">#<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/></xsl:attribute>
          &lt;<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>&gt;</a>
      </li>
      <!-- elements from this class -->
      <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice">
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      </xsl:apply-templates>
    </ul>
  </xsl:template>



  <!-- child elements -->
  <xsl:template mode="SIMPLECONTENT" match="xs:complexType">
    <xsl:if test="xs:sequence|xs:choice">
      <ul class="elementchild">
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice"/>
      </ul>
    </xsl:if>
  </xsl:template>

  <!-- occurance -->
  <xsl:template mode="OCCURANCE" match="xs:sequence|xs:choice|xs:element">
    <xsl:param name="ELEMENTNAME"/>
    <xsl:param name="COLORSTYLE"/>
    <xsl:if test="@minOccurs|@maxOccurs">
      <xsl:element name="{$ELEMENTNAME}">
        <xsl:attribute name="class">
          <xsl:value-of select="$COLORSTYLE"/>
        </xsl:attribute>
        <xsl:if test="@minOccurs=0 and not(@maxOccurs)">
          <span class="occurance">[optional]</span>
        </xsl:if>
        <xsl:if test="not(@minOccurs=0 and not(@maxOccurs))">
          <span class="occurance">[<xsl:if test="@minOccurs"><xsl:value-of select="@minOccurs"/></xsl:if><xsl:if test="not(@minOccurs)">1</xsl:if>-<xsl:if test="@maxOccurs"><xsl:value-of select="@maxOccurs"/></xsl:if><xsl:if test="not(@maxOccurs)">1</xsl:if>]</span>
        </xsl:if>
      </xsl:element>
    </xsl:if>
  </xsl:template>

  <!-- sequence -->
  <xsl:template mode="SIMPLECONTENT" match="xs:sequence">
    <xsl:param name="CLASSNAME"/>
    <li>
      <ul class="elementsequence">
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="ELEMENTNAME" select="'li'"/>
          <xsl:with-param name="COLORSTYLE" select="'elementsequencecolor'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </li>
  </xsl:template>

  <!-- choice -->
  <xsl:template mode="SIMPLECONTENT" match="xs:choice">
    <xsl:param name="CLASSNAME"/>
    <li>
      <ul class="elementchoice">
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="ELEMENTNAME" select="'li'"/>
          <xsl:with-param name="COLORSTYLE" select="'elementchoicecolor'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </li>
  </xsl:template>

  <!-- element -->
  <xsl:template mode="SIMPLECONTENT" match="xs:element">
    <xsl:param name="FUNCTIONNAME" select="@name"/>
    <xsl:param name="CLASSNAME"/>
    <li>
      <!-- name by not(ref) -->
      <xsl:if test="not(@ref)">
        <span class="element">&lt;<xsl:value-of select="@name"/>&gt;</span>
      </xsl:if>
      <!-- name by ref -->
      <xsl:if test="@ref">
        <a class="element">
          <xsl:attribute name="href">#<xsl:value-of select="@ref"/></xsl:attribute>
          &lt;<xsl:value-of select="@ref"/>&gt;</a>
      </xsl:if><xsl:text> </xsl:text>
      <!-- occurence -->
      <xsl:apply-templates mode="OCCURANCE" select=".">
        <xsl:with-param name="ELEMENTNAME" select="'span'"/>
      </xsl:apply-templates>
      <!-- type -->
      <xsl:if test="@type">
        <!-- type {http://openmbv.berlios.de/MBXMLUTILS/physicalvariable}* -->
        <xsl:if test="substring(@type,1,3)='pv:'">
          (Type: <a class="type">
             <!-- set href to $PHYSICALVARIABLEHTMLDOC#[scalartype|vectortype|matrixtype] -->
            <xsl:attribute name="href"><xsl:value-of select="$PHYSICALVARIABLEHTMLDOC"/>#<xsl:if test="substring(@type,string-length(@type)-5,6)='Scalar'">scalartype</xsl:if>
              <xsl:if test="substring(@type,string-length(@type)-5,6)='Vector'">vectortype</xsl:if>
              <xsl:if test="substring(@type,string-length(@type)-5,6)='Matrix'">matrixtype</xsl:if>
            </xsl:attribute>
            <xsl:value-of select="@type"/>
          </a>)
        </xsl:if>
        <!-- type not {http://openmbv.berlios.de/MBXMLUtils/physicalvariable}* -->
        <xsl:if test="substring(@type,1,3)!='pv:'">
          (Type: <span class="type">
            <xsl:value-of select="@type"/>
          </span>)
        </xsl:if>
      </xsl:if>
      <!-- element attributes -->
      <xsl:if test="@name and not(@type)">
        <xsl:apply-templates mode="ELEMENTATTRIBUTE" select="xs:complexType/xs:attribute"/>
      </xsl:if>
      <!-- documentation -->
      <xsl:apply-templates mode="ELEMENTANNOTATION" select="xs:annotation/xs:documentation"/>
      <!-- children -->
      <xsl:if test="@name and not(@type)">
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:complexType"/>
      </xsl:if>
    </li>
  </xsl:template>

  <!-- element attributes -->
  <xsl:template mode="ELEMENTATTRIBUTE" match="xs:attribute">
    <br/>
    <span class="attribute"><xsl:value-of select="@name"/></span>
    <xsl:if test="@use='required'">
      <span class="occurance"> [required]</span><xsl:text> </xsl:text>
    </xsl:if>
    <xsl:if test="@use!='required'">
      <span class="occurance"> [optional]</span><xsl:text> </xsl:text>
    </xsl:if>
    (Type: <span class="type"><xsl:value-of select="@type"/></span>)
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="CLASSANNOTATION" match="xs:annotation/xs:documentation">
    <xsl:if test="@source!='doxygen' or not(@source) or $INCLUDEDOXYGEN='true'">
      <xsl:if test="@source='doxygen'">
        <div class="classdocu"><b>The following part is the C++ API docucmentation from Doxygen</b></div>
      </xsl:if>
      <xsl:apply-templates mode="CLONEDOC"/>
    </xsl:if>
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="ELEMENTANNOTATION" match="xs:annotation/xs:documentation">
    <xsl:if test="@source!='doxygen' or not(@source) or $INCLUDEDOXYGEN='true'">
      <xsl:if test="@source='doxygen'">
        <div class="elementdocu"><b>The following part is the C++ API docucmentation from Doxygen</b></div>
      </xsl:if>
      <div class="elementdocu">
        <xsl:apply-templates mode="CLONEDOC"/>
      </div>
    </xsl:if>
  </xsl:template>

  <!-- clone doxygen xml/html part -->
  <xsl:template mode="CLONEDOC" match="*">
    <xsl:copy>
      <xsl:for-each select="@*">
        <xsl:copy/>
      </xsl:for-each>
      <xsl:apply-templates mode="CLONEDOC"/>
    </xsl:copy>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="text()">
    <xsl:copy/>
  </xsl:template>

  <xsl:template mode="CLONEDOC" match="html:object[@class='latexfigure']"/>

</xsl:stylesheet>
