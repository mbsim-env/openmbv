<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema"
  xmlns:html="http://www.w3.org/1999/xhtml"
  version="1.0">

  <!-- If changes in this file are made, then the analog changes must
       be done in the file xstToHtml.xsl -->

  <xsl:param name="PROJECT"/>



  <!-- output method -->
  <xsl:output method="text"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <xsl:template match="/">
<!-- tex header -->
\documentclass[a4]{report}
\usepackage{color}
\usepackage{graphicx}
\usepackage[utf8]{inputenc}
\usepackage{colortbl}
\usepackage{longtable}
\usepackage{tabularx}
\usepackage{titlesec}
\usepackage{titletoc}
\usepackage{listings}
\usepackage{amsmath,amssymb,amsbsy}
\usepackage[dvips]{hyperref}
\setlength{\parskip}{1em}
\setlength{\parindent}{0mm}

\pagestyle{headings}
\setlength{\hoffset}{-25.4mm}
\setlength{\oddsidemargin}{30mm}
\setlength{\evensidemargin}{30mm}
\setlength{\textwidth}{153mm}
\setlength{\voffset}{-35mm}
\setlength{\topmargin}{25mm}
\setlength{\headheight}{5mm}
\setlength{\headsep}{8mm}
\setlength{\textheight}{235.5mm}
\setlength{\footskip}{13mm}

% BEGIN format existing section/subsection/... and add/format new subparagraph[A-E]
\titleformat{\section}[hang]{\Large\bf}{\thesection}{1em}{}
\titleformat{\subsection}[hang]{\Large\bf}{\thesubsection}{1em}{}
\titleformat{\subsubsection}[hang]{\Large\bf}{\thesubsubsection}{1em}{}
\titleformat{\paragraph}[hang]{\Large\bf}{\theparagraph}{1em}{}
\titleformat{\subparagraph}[hang]{\Large\bf}{\thesubparagraph}{1em}{}
\titlespacing*{\section}{0ex}{12ex}{1ex}
\titlespacing*{\subsection}{0ex}{12ex}{1ex}
\titlespacing*{\subsubsection}{0ex}{12ex}{1ex}
\titlespacing*{\paragraph}{0ex}{12ex}{1ex}
\titlespacing*{\subparagraph}{0ex}{12ex}{1ex}

\titleclass{\subparagraphA}{straight}[\subparagraph]
\newcounter{subparagraphA}
\renewcommand{\thesubparagraphA}{\thesubparagraph.\arabic{subparagraphA}}
\titleformat{\subparagraphA}[hang]{\Large\bf}{\thesubparagraphA}{1em}{}
\titlespacing*{\subparagraphA}{0ex}{12ex}{1ex}

\titleclass{\subparagraphB}{straight}[\subparagraphA]
\newcounter{subparagraphB}
\renewcommand{\thesubparagraphB}{\thesubparagraphA.\arabic{subparagraphB}}
\titleformat{\subparagraphB}[hang]{\Large\bf}{\thesubparagraphB}{1em}{}
\titlespacing*{\subparagraphB}{0ex}{12ex}{1ex}

\titleclass{\subparagraphC}{straight}[\subparagraphB]
\newcounter{subparagraphC}
\renewcommand{\thesubparagraphC}{\thesubparagraphB.\arabic{subparagraphC}}
\titleformat{\subparagraphC}[hang]{\Large\bf}{\thesubparagraphC}{1em}{}
\titlespacing*{\subparagraphC}{0ex}{12ex}{1ex}

\titleclass{\subparagraphD}{straight}[\subparagraphC]
\newcounter{subparagraphD}
\renewcommand{\thesubparagraphD}{\thesubparagraphC.\arabic{subparagraphD}}
\titleformat{\subparagraphD}[hang]{\Large\bf}{\thesubparagraphD}{1em}{}
\titlespacing*{\subparagraphD}{0ex}{12ex}{1ex}

\titleclass{\subparagraphE}{straight}[\subparagraphD]
\newcounter{subparagraphE}
\renewcommand{\thesubparagraphE}{\thesubparagraphD.\arabic{subparagraphE}}
\titleformat{\subparagraphE}[hang]{\Large\bf}{\thesubparagraphE}{1em}{}
\titlespacing*{\subparagraphE}{0ex}{12ex}{1ex}

\dottedcontents{subsection}[1.5em]{}{0em}{0.75pc}
\dottedcontents{subsubsection}[3.0em]{}{0em}{0.75pc}
\dottedcontents{paragraph}[4.5em]{}{0em}{0.75pc}
\dottedcontents{subparagraph}[6.0em]{}{0em}{0.75pc}
\dottedcontents{subparagraphA}[7.5em]{}{0em}{0.75pc}
\dottedcontents{subparagraphB}[9.0em]{}{0em}{0.75pc}
\dottedcontents{subparagraphC}[10.5em]{}{0em}{0.75pc}
\dottedcontents{subparagraphD}[12.0em]{}{0em}{0.75pc}
\dottedcontents{subparagraphE}[13.5em]{}{0em}{0.75pc}
% END format existing section/subsection/... and add/format new subparagraph[A-E]

\setcounter{secnumdepth}{1}
\setcounter{tocdepth}{10}
\begin{document}

\title{<xsl:value-of select="$PROJECT"/> - XML Documentation\\[1cm]
  \normalsize{XML-Namespace: \textit{<xsl:value-of select="/xs:schema/@targetNamespace"/>}}}
\maketitle

\tableofcontents
\chapter{Introduction}
    <xsl:apply-templates mode="CLASSANNOTATION" select="/xs:schema/xs:annotation/xs:documentation"/>

\chapter{Nomenclature}
\label{nomenclature}

\section{An element}
\begin{tabular}{l}
  \lstinline[basicstyle=\bf\ttfamily]|&lt;ElementName&gt;| \textit{[0-2]} (Type: \lstinline[basicstyle=\ttfamily]|elementType|)\\
  \hspace{2ex}\lstinline[basicstyle=\bf\ttfamily]|attrName1| \textit{[required]} (Type: \lstinline[basicstyle=\ttfamily]|typeOfTheAttribute|)\\
  \hspace{2ex}\lstinline[basicstyle=\bf\ttfamily]|attrName2| \textit{[optional]} (Type: \lstinline[basicstyle=\ttfamily]|typeOfTheAttribute|)\\
  \hspace{3ex}
  \begin{minipage}[b]{\linewidth}
    Documentation of the element.
  \end{minipage}
\end{tabular}

The upper nomenclature defines a XML element named \lstinline[basicstyle=\bf\ttfamily]|ElementName| with (if given) a minimal occurance of 0 and a maximal occurance of 2. The element is of type \lstinline[basicstyle=\ttfamily]|elementType|.\\
A occurance of \textit{[optional]} means \textit{[0-1]}.\\
The element has two attributes named \lstinline[basicstyle=\bf\ttfamily]|attrName1| and \lstinline[basicstyle=\bf\ttfamily]|attrName2| of type \lstinline[basicstyle=\ttfamily]|typeOfTheAttribute|. A attribute can be optional or required.

\section{A choice of elment}
\begin{tabular}{!{\color{blue}\vline}@{\hspace{0.5pt}}l}
  {\color{blue}\textit{[1-2]}}\\
  \lstinline[basicstyle=\bf\ttfamily]|&lt;ElementA&gt;|\\
  \lstinline[basicstyle=\bf\ttfamily]|&lt;ElementB&gt;|\\
\end{tabular}

The upper nomenclature defines a choice of elements. Only one element of the given ones can be used. The choice has, if given, a minimal occurance of 1 and a maximal maximal occurence of 2.\\
A occurance of \textit{[optional]} means \textit{[0-1]}.

\section{A sequence of elments}
\begin{tabular}{!{\color{red}\vline}@{\hspace{0.5pt}}l}
  {\color{red}\textit{[1-2]}}\\
  \lstinline[basicstyle=\bf\ttfamily]|&lt;ElementA&gt;|\\
  \lstinline[basicstyle=\bf\ttfamily]|&lt;ElementB&gt;|\\
\end{tabular}

The upper nomenclature defines a sequence of elements. Each element must be given in that order. The sequence has, if given, a minimal occurance of 0 and a maximal maximal occurence of 3.\\
A occurance of \textit{[optional]} means \textit{[0-1]}.

\section{Nested sequences/choices}
\begin{tabular}{!{\color{red}\vline}@{\hspace{0.5pt}}l}
  {\color{red}\textit{[1-2]}}\\
  \lstinline[basicstyle=\bf\ttfamily]|&lt;ElementA&gt;|\\
  \begin{tabular}{!{\color{blue}\vline}@{\hspace{0.5pt}}l}
    {\color{blue}\textit{[0-3]}}\\
    \lstinline[basicstyle=\bf\ttfamily]|&lt;ElementC&gt;|\\
    \lstinline[basicstyle=\bf\ttfamily]|&lt;ElementD&gt;|\\
  \end{tabular}\\
  \lstinline[basicstyle=\bf\ttfamily]|&lt;ElementB&gt;|\\
\end{tabular}

Sequences and choices can be nested like above.

\section{Child Elements}
\begin{tabular}{!{\color{red}\vline}@{\hspace{0.5pt}}l}
  {\color{red}\textit{[1-2]}}\\
  \lstinline[basicstyle=\bf\ttfamily]|&lt;ParentElement&gt;|\\
  \hspace{5ex}
  \begin{tabular}{!{\color{blue}\vline}@{\hspace{0.5pt}}l}
    {\color{blue}\textit{[0-3]}}\\
    \lstinline[basicstyle=\bf\ttfamily]|&lt;ChildElementA&gt;|\\
    \lstinline[basicstyle=\bf\ttfamily]|&lt;ChildElementB&gt;|\\
  \end{tabular}\\
\end{tabular}

A indent indicates child elements for a given element.

\chapter{Elements}
    <xsl:for-each select="/xs:schema/xs:element/@substitutionGroup[not(.=/xs:schema/xs:element/@name) and not(.=preceding::*/@substitutionGroup)]">
      <xsl:sort select="."/>
      \subsection{\lstinline[basicstyle=\bf\ttfamily]|&lt;<xsl:value-of select="."/>&gt;|}
      This element is defined by the XML Schema (Project) with the namespace
      \textit{<xsl:value-of select="../namespace::*[name()=substring-before(current(),':')]"/>}, which is included
      by this XML Schema (Project). See the documentation of the included XML Schema (Project) for this element.
      <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[@substitutionGroup=current()]">
        <xsl:with-param name="LEVEL" select="1"/>
        <xsl:with-param name="LEVELNR" select="'3'"/>
        <xsl:sort select="@name"/>
      </xsl:apply-templates>
    </xsl:for-each>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[not(@substitutionGroup)]">
      <xsl:with-param name="LEVEL" select="0"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

\chapter{Simple Types}
    <xsl:apply-templates mode="SIMPLETYPE" select="/xs:schema/xs:simpleType">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

\end{document}
  </xsl:template>

  <!-- walk throw all elements -->
  <xsl:template mode="WALKCLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="NAME" select="@name"/>
    <xsl:apply-templates mode="CLASS" select=".">
      <xsl:with-param name="LEVEL" select="$LEVEL"/>
    </xsl:apply-templates>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[@substitutionGroup=$NAME]">
      <xsl:with-param name="LEVEL" select="$LEVEL+1"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>
  </xsl:template>

  <!-- class -->
  <xsl:template mode="CLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="TYPENAME" select="@type"/>
    <xsl:param name="CLASSNAME" select="@name"/>
    <!-- heading -->
    <xsl:choose>
      <!-- section is not used to prevent numbers on top level elements -->
      <xsl:when test='$LEVEL=0'>\subsection</xsl:when>
      <xsl:when test='$LEVEL=1'>\subsubsection</xsl:when>
      <xsl:when test='$LEVEL=2'>\paragraph</xsl:when>
      <xsl:when test='$LEVEL=3'>\subparagraph</xsl:when>
      <xsl:when test='$LEVEL=4'>\subparagraphA</xsl:when>
      <xsl:when test='$LEVEL=5'>\subparagraphB</xsl:when>
      <xsl:when test='$LEVEL=6'>\subparagraphC</xsl:when>
      <xsl:when test='$LEVEL=7'>\subparagraphD</xsl:when>
      <xsl:otherwise>\subparagraphE</xsl:otherwise>
    </xsl:choose>{\lstinline[basicstyle=\bf\ttfamily]|&lt;<xsl:value-of select="@name"/>&gt;|}
    \label{<xsl:value-of select="@name"/>}
    \makebox{}\\
    \setlength{\arrayrulewidth}{0.5pt}
    \begin{tabularx}{\textwidth}{|l|X|}
      \hline
      <!-- abstract -->
      Abstract Element: &amp;
      <xsl:if test="@abstract='true'">true</xsl:if>
      <xsl:if test="@abstract!='true' or not(@abstract)">false</xsl:if>\\
      \hline
      <!-- inherits -->
      Inherits: &amp;
      <xsl:if test="@substitutionGroup">
        \hyperref[<xsl:value-of select="@substitutionGroup"/>]{\lstinline[basicstyle=\bf\ttfamily]|&lt;<xsl:value-of select="@substitutionGroup"/>&gt;|}
        (P. \pageref*{<xsl:value-of select="@substitutionGroup"/>})
      </xsl:if>\\
      \hline
      <!-- inherited by -->
      Inherited by:
      <xsl:for-each select="/xs:schema/xs:element[@substitutionGroup=$CLASSNAME]">
        <xsl:sort select="@name"/>&amp; \hyperref[<xsl:value-of select="@name"/>]{\lstinline[basicstyle=\bf\ttfamily]|&lt;<xsl:value-of select="@name"/>&gt;|}~(P.~\pageref*{<xsl:value-of select="@name"/>}) \\
      </xsl:for-each>
      <xsl:if test="count(/xs:schema/xs:element[@substitutionGroup=$CLASSNAME])=0"> &amp; \\</xsl:if>
      \hline
      <!-- used in -->
      <!--Can be used in: &amp;
      <xsl:apply-templates mode="USEDIN2" select="."/>\\
      \hline-->
      <!-- class attributes -->
      Attributes: 
      <xsl:apply-templates mode="CLASSATTRIBUTE" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:attribute|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension/xs:attribute"/>
      <xsl:if test="not(/xs:schema/xs:complexType[@name=$TYPENAME]/xs:attribute)"> &amp; \\</xsl:if>
      \hline
    \end{tabularx}
    \setlength{\arrayrulewidth}{1.25pt}
    <!-- class documentation -->
    <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
    <!-- child elements -->
    <xsl:text>

      Child Elements:
       
    </xsl:text>
    <!-- child elements for not base class -->
    <xsl:apply-templates mode="CLASS" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension">
      <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
    </xsl:apply-templates>
    <!-- child elements for base class -->
    <xsl:if test="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
      <xsl:apply-templates mode="SIMPLECONTENT" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        <xsl:with-param name="FIRST" select="'true'"/>
      </xsl:apply-templates>
    </xsl:if>
  </xsl:template>

  <!-- simple type -->
  <xsl:template mode="SIMPLETYPE" match="/xs:schema/xs:simpleType">
    \subsection{\lstinline[basicstyle=\bf\ttfamily]|<xsl:value-of select="@name"/>|}
    \label{<xsl:value-of select="@name"/>}
    <!-- simpleType documentation -->
    <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
  </xsl:template>

  <!-- class attributes -->
  <xsl:template mode="CLASSATTRIBUTE" match="/xs:schema/xs:complexType/xs:attribute|/xs:schema/xs:complexType/xs:complexContent/xs:extension/xs:attribute">
    <xsl:if test="@name">
      &amp; \lstinline[basicstyle=\bf\ttfamily]|<xsl:value-of select="@name"/>|
      <xsl:if test="@use='required'">\textit{[required]}</xsl:if>
      <xsl:if test="@use!='required' or not(@use)">\textit{[optional]}</xsl:if>
      (Type: \hyperref[<xsl:value-of select="@type"/>]{\lstinline[basicstyle=\ttfamily]|<xsl:value-of select="@type"/>|})\\
    </xsl:if>
    <xsl:if test="@ref">
      &amp; \hyperref[<xsl:value-of select="@ref"/>]{\lstinline[basicstyle=\bf\ttfamily]|<xsl:value-of select="@ref"/>|}\\
    </xsl:if>
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
    <xsl:param name="CLASSTYPE" select="@name"/>\hyperref[<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>]{\lstinline[basicstyle=\bf\ttfamily]|&lt;<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>&gt;|}~(P.~\pageref*{<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>}),
  </xsl:template>-->

  <!-- child elements for not base class -->
  <xsl:template mode="CLASS" match="xs:extension">
    <xsl:param name="CLASSNAME"/>
    <!-- elements from base class -->
    All Elements from \hyperref[<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>]{\lstinline[basicstyle=\bf\ttfamily]|&lt;<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>&gt;|}~(P.~\pageref*{<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>})\\
    <!-- elements from this class -->
    <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice">
      <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      <xsl:with-param name="FIRST" select="'true'"/>
    </xsl:apply-templates>
  </xsl:template>



  <!-- child elements -->
  <xsl:template mode="SIMPLECONTENT" match="xs:complexType">
    <xsl:if test="xs:sequence|xs:choice">
      \hspace{5ex}
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice"/>
    </xsl:if>
  </xsl:template>

  <!-- occurance -->
  <xsl:template mode="OCCURANCE" match="xs:sequence|xs:choice|xs:element">
    <xsl:param name="ELEMENTNAME"/>
    <xsl:param name="COLOR"/>
    <xsl:if test="@minOccurs|@maxOccurs">
      {<xsl:if test="$COLOR!=''">\color{<xsl:value-of select="$COLOR"/>}</xsl:if>
      <xsl:if test="@minOccurs=0 and not(@maxOccurs)">
        \textit{[optional]}
      </xsl:if>
      <xsl:if test="not(@minOccurs=0 and not(@maxOccurs))">
        \textit{[<xsl:if test="@minOccurs"><xsl:value-of select="@minOccurs"/></xsl:if><xsl:if test="not(@minOccurs)">1</xsl:if>-<xsl:if test="@maxOccurs"><xsl:value-of select="@maxOccurs"/></xsl:if><xsl:if test="not(@maxOccurs)">1</xsl:if>]}
      </xsl:if>}
      <xsl:if test="$ELEMENTNAME='li'">
        \\
      </xsl:if>
    </xsl:if>
  </xsl:template>

  <!-- sequence -->
  <xsl:template mode="SIMPLECONTENT" match="xs:sequence">
    <xsl:param name="CLASSNAME"/>
    <xsl:param name="FIRST"/>
    <xsl:if test="$FIRST='true'">\begin{longtable}[l]{!{\color{red}\vline}@{\hspace{0.5pt}}l}</xsl:if>
    <xsl:if test="$FIRST!='true'">\begin{tabular}{!{\color{red}\vline}@{\hspace{0.5pt}}l}</xsl:if>
      <xsl:apply-templates mode="OCCURANCE" select=".">
        <xsl:with-param name="ELEMENTNAME" select="'li'"/>
        <xsl:with-param name="COLOR" select="'red'"/>
      </xsl:apply-templates>
      <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      </xsl:apply-templates>
    <xsl:if test="$FIRST='true'">\end{longtable}</xsl:if>
    <xsl:if test="$FIRST!='true'">\end{tabular}\\[-2ex]\\</xsl:if>
  </xsl:template>

  <!-- choice -->
  <xsl:template mode="SIMPLECONTENT" match="xs:choice">
    <xsl:param name="CLASSNAME"/>
    <xsl:param name="FIRST"/>
    <xsl:if test="$FIRST='true'">\begin{longtable}[l]{!{\color{blue}\vline}@{\hspace{0.5pt}}l}</xsl:if>
    <xsl:if test="$FIRST!='true'">\begin{tabular}{!{\color{blue}\vline}@{\hspace{0.5pt}}l}</xsl:if>
      <xsl:apply-templates mode="OCCURANCE" select=".">
        <xsl:with-param name="ELEMENTNAME" select="'li'"/>
        <xsl:with-param name="COLOR" select="'blue'"/>
      </xsl:apply-templates>
      <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      </xsl:apply-templates>
    <xsl:if test="$FIRST='true'">\end{longtable}</xsl:if>
    <xsl:if test="$FIRST!='true'">\end{tabular}\\[-2ex]\\</xsl:if>
  </xsl:template>

  <!-- element -->
  <xsl:template mode="SIMPLECONTENT" match="xs:element">
    <xsl:param name="FUNCTIONNAME" select="@name"/>
    <xsl:param name="CLASSNAME"/>
    <!-- name by not(ref) -->
    <xsl:if test="not(@ref)">\lstinline[basicstyle=\bf\ttfamily]|&lt;<xsl:value-of select="@name"/>&gt;|</xsl:if>
    <!-- name by ref -->
    <xsl:if test="@ref">\hyperref[<xsl:value-of select="@ref"/>]{\lstinline[basicstyle=\bf\ttfamily]|&lt;<xsl:value-of select="@ref"/>&gt;|}
      (P. \pageref*{<xsl:value-of select="@ref"/>})</xsl:if>
    <!-- occurence -->
    <xsl:apply-templates mode="OCCURANCE" select="."><xsl:with-param name="ELEMENTNAME" select="'span'"/></xsl:apply-templates>
    <!-- type -->
    <xsl:if test="@type">(Type: \hyperref[<xsl:value-of select="@type"/>]{\lstinline[basicstyle=\ttfamily]|<xsl:value-of select="@type"/>|})</xsl:if>\\
    <!-- element attributes -->
    <xsl:if test="@name and not(@type)">
      <xsl:if test="xs:complexType/xs:attribute">
        <xsl:apply-templates mode="ELEMENTATTRIBUTE" select="xs:complexType/xs:attribute"/>
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
    \hspace{2ex}\lstinline[basicstyle=\bf\ttfamily]|<xsl:value-of select="@name"/>|
    <xsl:if test="@use='required'"> \textit{[required]}</xsl:if>
    <xsl:if test="@use!='required'"> \textit{[optional]}</xsl:if>
    (Type: \hyperref[<xsl:value-of select="@type"/>]{\lstinline[basicstyle=\ttfamily]|<xsl:value-of select="@type"/>|})\\
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="CLASSANNOTATION" match="xs:annotation/xs:documentation">
    <!-- print info about Doxygen documentation only if other docu exist -->
    <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])>0"><xsl:text>

      \textbf{Doxygen documentation}

    </xsl:text></xsl:if><xsl:text>

</xsl:text><xsl:apply-templates mode="CLONEDOC"/>
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="ELEMENTANNOTATION" match="xs:annotation/xs:documentation">
    \hspace{3ex}
    \begin{minipage}[b]{\linewidth}
      <!-- print info about Doxygen documentation only if other docu exist -->
      <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])>0"><xsl:text>

        \textbf{Doxygen documentation}

      </xsl:text></xsl:if>
      <xsl:apply-templates mode="CLONEDOC"/>
    \end{minipage}\\[1ex]
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

  <xsl:template mode="CLONEDOC" match="text()"><xsl:text> </xsl:text><xsl:value-of select="normalize-space(.)"/></xsl:template>

  <xsl:template mode="CLONEDOC" match="html:div[@class='para']|html:p"><xsl:text>

</xsl:text><xsl:apply-templates mode="CLONEDOC"/><xsl:text>

</xsl:text></xsl:template>

  <xsl:template mode="CLONEDOC" match="html:br"><xsl:text> \\
</xsl:text></xsl:template>

  <xsl:template mode="CLONEDOC" match="html:dl"><xsl:text>\begin{description}
</xsl:text><xsl:apply-templates mode="CLONEDOC"/><xsl:text>\end{description}
</xsl:text></xsl:template>
  <xsl:template mode="CLONEDOC" match="html:dt"><xsl:text>\item[</xsl:text><xsl:value-of select="."/><xsl:text>]
</xsl:text></xsl:template>
  <xsl:template mode="CLONEDOC" match="html:dd"><xsl:apply-templates mode="CLONEDOC"/></xsl:template>

  <xsl:template mode="CLONEDOC" match="html:ul"><xsl:text>\begin{itemize}
</xsl:text><xsl:apply-templates mode="CLONEDOC"/><xsl:text>\end{itemize}
</xsl:text></xsl:template>
  <xsl:template mode="CLONEDOC" match="html:li"><xsl:text>\item
</xsl:text><xsl:apply-templates mode="CLONEDOC"/></xsl:template>

  <xsl:template mode="CLONEDOC" match="html:ol"><xsl:text>\begin{enumerate}
</xsl:text><xsl:apply-templates mode="CLONEDOC"/><xsl:text>\end{enumerate}
</xsl:text></xsl:template>
  <xsl:template mode="CLONEDOC" match="html:li"><xsl:text>\item
</xsl:text><xsl:apply-templates mode="CLONEDOC"/></xsl:template>

  <xsl:template mode="CLONEDOC" match="html:b">\textbf{<xsl:apply-templates mode="CLONEDOC"/>}</xsl:template>

  <xsl:template mode="CLONEDOC" match="html:i">\textit{<xsl:apply-templates mode="CLONEDOC"/>}</xsl:template>

  <xsl:template mode="CLONEDOC" match="html:tt">\lstinline[basicstyle=\bf\ttfamily]|<xsl:apply-templates mode="CLONEDOC"/>|</xsl:template>

  <xsl:template mode="CLONEDOC" match="html:pre">\begin{verbatim}<xsl:value-of select="."/>\end{verbatim}</xsl:template>

  <xsl:template mode="CLONEDOC" match="html:object[@class='eqn']"><xsl:text>
\[ </xsl:text><xsl:value-of select="."/><xsl:text> \]
</xsl:text></xsl:template>
  <xsl:template mode="CLONEDOC" match="html:object[@class='inlineeqn']"> $<xsl:value-of select="."/>$ </xsl:template>

  <xsl:template mode="CLONEDOC" match="html:object[@class='figure']"><xsl:text>
\begin{center}
  \includegraphics[width=</xsl:text><xsl:value-of select="@standby"/>]{<xsl:value-of select="@data"/><xsl:text>.eps}\\
</xsl:text><xsl:value-of select="@title"/><xsl:text>
\end{center}</xsl:text></xsl:template>
  <xsl:template mode="CLONEDOC" match="html:object[@class='figure_latex']"><xsl:text>
\begin{center}
  \includegraphics[width=</xsl:text><xsl:value-of select="@standby"/>]{<xsl:value-of select="@data"/><xsl:text>}\\
</xsl:text><xsl:value-of select="@title"/><xsl:text>
\end{center}</xsl:text></xsl:template>
  <xsl:template mode="CLONEDOC" match="html:object[@class='figure_html']"></xsl:template>
  <xsl:template mode="CLONEDOC" match="html:a[@class='link']">
    <xsl:if test="contains(@href,':')">\hyperref[<xsl:value-of select="substring-after(@href,':')"/>]{<xsl:apply-templates mode="CLONEDOC"/>}</xsl:if>
    <xsl:if test="not(contains(@href,':'))">\hyperref[<xsl:value-of select="@href"/>]{<xsl:apply-templates mode="CLONEDOC"/>}</xsl:if>
  </xsl:template>
 
</xsl:stylesheet>
