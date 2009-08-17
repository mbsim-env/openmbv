<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema"
  xmlns:html="http://www.w3.org/1999/xhtml"
  version="1.0">

  <xsl:param name="PROJECT"/>
  <xsl:param name="PHYSICALVARIABLEHTMLDOC"/>

  <xsl:template name="CONUNDERSCORE">
    <xsl:param name="V"/>
    <xsl:if test="not(contains($V,'_'))">
      <xsl:value-of select="$V"/>
    </xsl:if>
    <xsl:if test="contains($V,'_')">
      <xsl:value-of select="concat(substring-before($V,'_'),'\_',substring-after($V,'_'))"/>
    </xsl:if>
  </xsl:template>



  <!-- output method -->
  <xsl:output method="text"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <xsl:template match="/">
<!-- tex header -->
\documentclass{report}
\usepackage{color}
\usepackage{graphicx}
\setlength{\parskip}{1em}
\setlength{\parindent}{0mm}
\begin{document}
\begin{center}
  {\Huge <xsl:value-of select="$PROJECT"/> - XML Documentation}

  This is the Documentation of the XML representation for <xsl:value-of select="$PROJECT"/>.
\end{center}
\tableofcontents
\chapter{Nomenclature}
\label{nomenclature}

\section{A element}
\begin{list}{}{\leftmargin=0.5em}
  \item[] \texttt{$&lt;$ElementName$&gt;$} X0-2X (Type: \texttt{elementType})
  \begin{list}{}{\leftmargin=1em}
    \item attrName1 XrequiredX (Type: \texttt{typeOfTheAttribute})
    \item attrName2 XoptionalX (Type: \texttt{typeOfTheAttribute})
  \end{list}
  Documentation of the element.
\end{list}
The upper nomenclature defines a XML element named \texttt{ElementName} with (if given) a minimal occurance of 0 and a maximal occurance of 2. The element is of type \texttt{elementType}.\\
A occurance of XoptionalX means X0-1X.\\
The element has two attributes named \texttt{attrName1} and \texttt{attrName2} of type \texttt{typeOfTheAttribute}. A attribute can be optional or required.

\section{A choice of elment}
\begin{list}{}{\leftmargin=0.5em}
  \color{blue}
  \item X1-2X
  \item \texttt{$&lt;$ElementA$&gt;$}
  \item \texttt{$&lt;$ElementB$&gt;$}
\end{list}
The upper nomenclature defines a choice of elements. Only one element of the given ones can be used. The choice has, if given, a minimal occurance of 1 and a maximal maximal occurence of 2.\\
A occurance of XoptionalX means X0-1X.

\section{A sequence of elments}
\begin{list}{}{\leftmargin=0.5em}
  \color{red}
  \item X0-3X
  \item \texttt{$&lt;$ElementA$&gt;$}
  \item \texttt{$&lt;$ElementB$&gt;$}
\end{list}
The upper nomenclature defines a sequence of elements. Each element must be given in that order. The sequence has, if given, a minimal occurance of 0 and a maximal maximal occurence of 3.\\
A occurance of XoptionalX means X0-1X.

\section{Nested sequences/choices}
\begin{list}{}{\leftmargin=0.5em}
  \color{red}
  \item X1-2X
  \item \texttt{$&lt;$ElementA$&gt;$}
  \item
    \begin{list}{}{\leftmargin=0.5em}
      \color{blue}
      \item X0-3X
      \item \texttt{$&lt;$ElementC$&gt;$}
      \item \texttt{$&lt;$ElementD$&gt;$}
    \end{list}
  \item \texttt{$&lt;$ElementB$&gt;$}
\end{list}
Sequences and choices can be nested like above.

\section{Child Elements}
\begin{list}{}{\leftmargin=0.5em}
  \color{red}
  \item X1-2X
  \item \texttt{$&lt;$ParentElement$&gt;$}
  \item
    \begin{list}{}{\leftmargin=3em}
      \color{blue}
      \item X0-3X
      \item \texttt{$&lt;$ChildElementA$&gt;$}
      \item \texttt{$&lt;$ChildElementB$&gt;$}
    \end{list}
\end{list}
A indent indicates child elements for a given element.

\chapter{Elements}
    <xsl:apply-templates mode="CLASS" select="/xs:schema/xs:element">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

\end{document}
  </xsl:template>

  <!-- class -->
  <xsl:template mode="CLASS" match="/xs:schema/xs:element">
    <xsl:param name="TYPENAME" select="@type"/>
    <xsl:param name="CLASSNAME" select="@name"/>
    <!-- heading -->
    \section{\texttt{$&lt;$<xsl:call-template name="CONUNDERSCORE"><xsl:with-param name="V" select="@name"/></xsl:call-template>$&gt;$}}
    \label{<xsl:value-of select="@name"/>}
    <!-- abstract -->
    <xsl:if test="@abstract='true'"><xsl:text>

      This element ist abstract.

    </xsl:text></xsl:if>
    <!-- inherits -->
    <xsl:if test="@substitutionGroup"><xsl:text>

      Inherits:</xsl:text>
      \texttt{$&lt;$<xsl:call-template name="CONUNDERSCORE"><xsl:with-param name="V" select="@substitutionGroup"/></xsl:call-template>$&gt;$}
      (\ref{<xsl:value-of select="@substitutionGroup"/>}, p. \pageref{<xsl:value-of select="@substitutionGroup"/>})

    </xsl:if>
    <!-- inherited by -->
    <xsl:if test="count(/xs:schema/xs:element[@substitutionGroup=$CLASSNAME])>0"><xsl:text>

      Inherited by:</xsl:text>
      <xsl:for-each select="/xs:schema/xs:element[@substitutionGroup=$CLASSNAME]">
        <xsl:sort select="@name"/>\texttt{$&lt;$<xsl:call-template name="CONUNDERSCORE"><xsl:with-param name="V" select="@name"/></xsl:call-template>$&gt;$}
        (\ref{<xsl:value-of select="@name"/>}, p. \pageref{<xsl:value-of select="@name"/>})
        <xsl:if test="position()!=last()">, </xsl:if>
      </xsl:for-each>

    </xsl:if>
    <!-- used in --><xsl:text>

    Can be used in: </xsl:text><xsl:apply-templates mode="USEDIN2" select="."/><xsl:text>

    </xsl:text>
    <!-- class documentation -->
    <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
    <!-- class attributes -->
    <xsl:apply-templates mode="CLASSATTRIBUTE" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:attribute"/>
    <!-- child elements -->
    <xsl:if test="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension/xs:choice|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice"><xsl:text>

      Child Elements:
       
      </xsl:text>
      <!-- child elements for not base class -->
      <xsl:apply-templates mode="CLASS" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension">
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      </xsl:apply-templates>
      <!-- child elements for base class -->
      <xsl:if test="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
        \begin{list}{}{\leftmargin=0.5em}
          <xsl:apply-templates mode="SIMPLECONTENT" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
            <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
          </xsl:apply-templates>
        \end{list}
      </xsl:if>
    </xsl:if>
  </xsl:template>

  <!-- class attributes -->
  <xsl:template mode="CLASSATTRIBUTE" match="/xs:schema/xs:complexType/xs:attribute"><xsl:text>

    </xsl:text>Attribute: \texttt{<xsl:value-of select="@name"/>}
    <xsl:if test="@use='required'">XrequiredX</xsl:if>
    <xsl:if test="@use!='required'">XoptionalX</xsl:if>
    (Type: \texttt{<xsl:value-of select="@type"/>})<xsl:text>

  </xsl:text></xsl:template>

  <!-- used in -->
  <xsl:template mode="USEDIN2" match="/xs:schema/xs:element">
    <xsl:param name="SUBSTGROUP" select="@substitutionGroup"/>
    <xsl:param name="CLASSNAME" select="@name"/>
    <xsl:apply-templates mode="USEDIN" select="/descendant::xs:element[@ref=$CLASSNAME]"/>
    <xsl:apply-templates mode="USEDIN2" select="/xs:schema/xs:element[@name=$SUBSTGROUP]"/>
  </xsl:template>
  <xsl:template mode="USEDIN" match="xs:element">
    <xsl:apply-templates mode="USEDIN" select="ancestor::xs:complexType[last()]"/>
  </xsl:template>
  <xsl:template mode="USEDIN" match="xs:complexType">
    <xsl:param name="CLASSTYPE" select="@name"/>\texttt{$&lt;$<xsl:call-template name="CONUNDERSCORE"><xsl:with-param name="V" select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/></xsl:call-template>$&gt;$}
    (\ref{<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>}, p. \pageref{<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>})
  </xsl:template>

  <!-- child elements for not base class -->
  <xsl:template mode="CLASS" match="xs:extension">
    <xsl:param name="CLASSNAME"/>
    <xsl:if test="xs:sequence|xs:choice">
      \begin{list}{}{\leftmargin=0.5em}
        <!-- elements from base class -->
        \item All Elements from \texttt{$&lt;$<xsl:call-template name="CONUNDERSCORE"><xsl:with-param name="V" select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/></xsl:call-template>$&gt;$}
          (\ref{<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>}, p. \pageref{<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>})
        <!-- elements from this class -->
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      \end{list}
    </xsl:if>
  </xsl:template>



  <!-- child elements -->
  <xsl:template mode="SIMPLECONTENT" match="xs:complexType">
    <xsl:if test="xs:sequence|xs:choice">
      \begin{list}{}{\leftmargin=3em}
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice"/>
      \end{list}
    </xsl:if>
  </xsl:template>

  <!-- occurance -->
  <xsl:template mode="OCCURANCE" match="xs:sequence|xs:choice|xs:element">
    <xsl:param name="ELEMENTNAME"/>
    <xsl:param name="COLOR"/>
    <xsl:if test="@minOccurs|@maxOccurs">
      <xsl:if test="@minOccurs=0 and not(@maxOccurs)">
        XoptionalX
      </xsl:if>
      <xsl:if test="not(@minOccurs=0 and not(@maxOccurs))">
        X<xsl:if test="@minOccurs"><xsl:value-of select="@minOccurs"/></xsl:if><xsl:if test="not(@minOccurs)">1</xsl:if>-<xsl:if test="@maxOccurs"><xsl:value-of select="@maxOccurs"/></xsl:if><xsl:if test="not(@maxOccurs)">1</xsl:if>X
      </xsl:if>
    </xsl:if>
  </xsl:template>

  <!-- sequence -->
  <xsl:template mode="SIMPLECONTENT" match="xs:sequence">
    <xsl:param name="CLASSNAME"/>
    \item
      \begin{list}{}{\leftmargin=0.5em}
        \color{red}
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="ELEMENTNAME" select="'li'"/>
          <xsl:with-param name="COLOR" select="'red'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      \end{list}
  </xsl:template>

  <!-- choice -->
  <xsl:template mode="SIMPLECONTENT" match="xs:choice">
    <xsl:param name="CLASSNAME"/>
    \item
      \begin{list}{}{\leftmargin=0.5em}
        \color{blue}
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="ELEMENTNAME" select="'li'"/>
          <xsl:with-param name="COLOR" select="'blue'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      \end{list}
  </xsl:template>

  <!-- element -->
  <xsl:template mode="SIMPLECONTENT" match="xs:element">
    <xsl:param name="FUNCTIONNAME" select="@name"/>
    <xsl:param name="CLASSNAME"/>
    \item
      <!-- name by not(ref) -->
      <xsl:if test="not(@ref)">\texttt{$&lt;$<xsl:call-template name="CONUNDERSCORE"><xsl:with-param name="V" select="@name"/></xsl:call-template>$&gt;$}</xsl:if>
      <!-- name by ref -->
      <xsl:if test="@ref">\texttt{$&lt;$<xsl:call-template name="CONUNDERSCORE"><xsl:with-param name="V" select="@ref"/></xsl:call-template>$&gt;$}
        (\ref{<xsl:value-of select="@ref"/>}, p. \pageref{<xsl:value-of select="@ref"/>})</xsl:if>
      <!-- occurence -->
      <xsl:apply-templates mode="OCCURANCE" select="."><xsl:with-param name="ELEMENTNAME" select="'span'"/></xsl:apply-templates>
      <!-- type -->
      <xsl:if test="@type">(Type: \texttt{<xsl:value-of select="@type"/>})</xsl:if>
      <!-- element attributes -->
      <xsl:if test="@name and not(@type)">
        <xsl:if test="xs:complexType/xs:attribute">
          \begin{list}{}{\leftmargin=1em}
            <xsl:apply-templates mode="ELEMENTATTRIBUTE" select="xs:complexType/xs:attribute"/>
          \end{list}
        </xsl:if>
      </xsl:if>
      <!-- documentation -->
      <xsl:apply-templates mode="ELEMENTANNOTATION" select="xs:annotation/xs:documentation"/>
      <!-- children -->
      <xsl:if test="@name and not(@type)">
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:complexType"/>
      </xsl:if>
  </xsl:template>

  <!-- element attributes -->
  <xsl:template mode="ELEMENTATTRIBUTE" match="xs:attribute">
    \item <xsl:value-of select="@name"/>
    <xsl:if test="@use='required'"> XrequiredX</xsl:if>
    <xsl:if test="@use!='required'"> XoptionalX</xsl:if> (Type: \texttt{<xsl:value-of select="@type"/>})
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="CLASSANNOTATION" match="xs:annotation/xs:documentation">
    <xsl:if test="@source='doxygen'"><xsl:text>

      The following part is the C++ API docucmentation from Doxygen

    </xsl:text></xsl:if>
    <xsl:apply-templates mode="CLONEDOC"/>
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="ELEMENTANNOTATION" match="xs:annotation/xs:documentation">
    \begin{list}{}{\leftmargin=5ex}
      \item
      <xsl:if test="@source='doxygen'"><xsl:text>

        </xsl:text>The following part is the C++ API docucmentation from Doxygen<xsl:text>

      </xsl:text></xsl:if>
      <xsl:apply-templates mode="CLONEDOC"/>
    \end{list}
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

  <xsl:template mode="CLONEDOC" match="html:div[@class='para']"><xsl:text>

    </xsl:text><xsl:apply-templates mode="CLONEDOC"/><xsl:text>

  </xsl:text></xsl:template>

  <xsl:template mode="CLONEDOC" match="html:dl">
    \begin{description}
      <xsl:apply-templates mode="CLONEDOC"/>
    \end{description}
  </xsl:template>

  <xsl:template mode="CLONEDOC" match="html:dt">
    \item[<xsl:value-of select="."/>]
  </xsl:template>

  <xsl:template mode="CLONEDOC" match="html:dd">
    <xsl:apply-templates mode="CLONEDOC"/>
  </xsl:template>

  <xsl:template mode="CLONEDOC" match="html:img[@class='eqn']|html:img[@class='inlineeqn']">
    <xsl:value-of select="@alt"/>
  </xsl:template>

  <xsl:template mode="CLONEDOC" match="html:div[@class='htmlfigure']"/>

  <xsl:template mode="CLONEDOC" match="html:object[@class='latexfigure']">
    \begin{center}\includegraphics[width=<xsl:value-of select="@standby"/>]{<xsl:value-of select="@data"/>}\\<xsl:value-of select="@title"/>\end{center}
  </xsl:template>
 
</xsl:stylesheet>
