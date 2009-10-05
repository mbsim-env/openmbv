<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:mm="http://openmbv.berlios.de/MBXMLUtils/measurement"
  xmlns="http://www.w3.org/1999/xhtml"
  version="1.0">

  <!-- If changes in this file are made, then the analog changes must
       be done in the file measurementToHtml.xsl -->

  <!-- output method -->
  <xsl:output method="text" encoding="UTF-8"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <!-- for all direct root child elements (classes) -->
  <xsl:template match="/">
    <!-- tex header -->
\documentclass[a4]{report}
\usepackage[utf8]{inputenc}
\usepackage[dvips]{hyperref}
\usepackage{bold-extra}
\usepackage{listings}
\setlength{\parskip}{1em}
\setlength{\parindent}{0mm}

\pagestyle{headings}
\setlength{\hoffset}{-25.4mm}
\setlength{\oddsidemargin}{30mm}
\setlength{\evensidemargin}{30mm}
\setlength{\textwidth}{153mm}
\setlength{\voffset}{-25.4mm}
\setlength{\topmargin}{25mm}
\setlength{\headheight}{5mm}
\setlength{\headsep}{8mm}
\setlength{\textheight}{235.5mm}
\setlength{\footskip}{13mm}

\begin{document}

\title{Physical Variable - XML Documentation}
\maketitle

\tableofcontents

\chapter{Introduction}
\label{introduction}

\chapter{Element Name}
\label{name}
Elements which must be referable must have a name. Mostly this name is given by the attribute \verb|name|. A valid name starts with a letter or a underscore. The following characters can be letters, underscores or digits. The content between '\verb|{|' and '\verb|}|' can be any \hyperref[octave]{Octave Expression/Program} and is substituted by the result of Octave (which must be a valid name; normal a integer number).

The following table shows examples for valid element names (on the left) and the substituted names (on the right), if there exist a scalar (integer) parameter of name \verb|n| with the value \verb|2|:
\begin{tabular}{|l|l|}
  \hline
  \textbf{Element Name} &amp; \textbf{Substituted Element Name}\\
  \hline
  \verb|Object1| &amp; \verb|Object1|\\
  \verb|myname_3| &amp; \verb|myname_3|\\
  \verb|body{n+6}| &amp; \verb|body8|\\
  \verb|Obj_{n+4}_{if n==2; ret=3; else ret=6; end}| &amp; \verb|Obj_6_3|\\
  \hline
\end{tabular}

\chapter{Scalar Type}
\label{scalartype}
A scalar type can be of any unit defined in \hyperref[measurement]{measurements}~(P.~\pageref*{measurement}). The unit is given by a optional attribute of name \texttt{unit}.
The type name of a scalar of measure length is \texttt{pv:lengthScalar} and so on.
Where \texttt{pv} is mapped to the namespace-uri\\
\texttt{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}.

The content of a scalar type must be a \hyperref[octave]{octave~expression/program}~(P.~\pageref*{octave}). The following examples are valid, if there exist a scalar paremter \verb|a| and \verb|b| in the \hyperref[parameters]{parameter~file}~(P.~\pageref*{parameters}):
\begin{verbatim}
  &lt;myScalarElement unit="mm"&gt;4*sin(a)+b&lt;/myScalarElement&gt;
  &lt;myScalarElement&gt;[a,2]*[3;b]&lt;/myScalarElement&gt;
\end{verbatim}

There is also a special unit of name \texttt{unknown} defined. This unit dose not take the optional \texttt{unit}
attribute, it takes an optional attribute of name \texttt{convertUnit}. The value of this attribute can be a
\hyperref[octave]{Octave Expression}~(P.~\pageref*{octave})
which must contain a parameter of name \texttt{value}. The given value is then converted by this expression.

\chapter{Vector Type}
\label{vectortype}
A vector type can be of any unit defined in \hyperref[measurement]{measurements}~(P.~\pageref*{measurement}). The unit is given by a optional attribute of name \texttt{unit}.
The type name of a vector of measure length is \texttt{pv:lengthVector} and so on.
Where \texttt{pv} is mapped to the namespace-uri\\
\texttt{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}.

The content of a vector type can be one of the following:
\begin{itemize}
  \item A \hyperref[octave]{octave~expression/program}~(P.~\pageref*{octave}). The following examples are valid, if there exist a scalar paremter \verb|a| and \verb|b| in the \hyperref[parameters]{parameter~file}~(P.~\pageref*{parameters}):
    \begin{verbatim}
&lt;myVectorElement unit="mm"&gt;[1;b;a;7]&lt;/myVectorElement&gt;
&lt;myVectorElement&gt;[a,2;5.6,7]*[3;b]&lt;/myVectorElement&gt;
    \end{verbatim}
    Using the octave load command it is also possible to load the data from a external file:
    \begin{verbatim}
&lt;myVectorElement unit="mm"&gt;ret=load('myfile.dat')&lt;/myVectorElement&gt;
    \end{verbatim}
  \item A XML representation of a vector: The following shows a example of such a XML representation.
    \begin{verbatim}
&lt;myVectorElement
  xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;7.3&lt;/pv:ele&gt;
&lt;/myVectorElement&gt;
    \end{verbatim}
\end{itemize}

For the special unit of name \texttt{unknown} see \hyperref[scalartype]{Scalar Type}~(P.~\pageref*{scalartype})

\chapter{Matrix Type}
\label{matrixtype}
A matrix type can be of any unit defined in \hyperref[measurement]{measurements}~(P.~\pageref*{measurement}). The unit is given by a optional attribute of name \texttt{unit}.
The type name of a matrix of measure length is \texttt{pv:lengthMatrix} and so on.
Where \texttt{pv} is mapped to the namespace-uri\\
\texttt{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}.

The content of a matrix type can be one of the following:
\begin{itemize}
  \item A \hyperref[octave]{octave~expression/program}~(P.~\pageref*{octave}). The following examples are valid, if there exist a scalar paremter \verb|a| and \verb|b| in the \hyperref[parameters]{parameter~file}~(P.~\pageref*{parameters}):
    \begin{verbatim}
&lt;myMatrixElement&gt;[1,b;a,7]&lt;/myMatrixElement&gt;
&lt;myMatrixElement&gt;[a,2;5.6,7]*rand(2,2)&lt;/myMatrixElement&gt;
    \end{verbatim}
    Using the octave load command it is also possible to load the data from a external file:
    \begin{verbatim}
&lt;myMatrixElement&gt;ret=load('mydata.dat')&lt;/myMatrixElement&gt;
    \end{verbatim}
  \item A XML representation of a matrix: The following shows a example of such a XML representation.
    \begin{verbatim}
&lt;myMatrixElement
  xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:row&gt;
    &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
    &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;/pv:row&gt;
  &lt;pv:row&gt;
    &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
    &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;/pv:row&gt;
&lt;/myMatrixElement&gt;
    \end{verbatim}
\end{itemize}

For the special unit of name \texttt{unknown} see \hyperref[scalartype]{Scalar Type}~(P.~\pageref*{scalartype})

\chapter{Parameters}
\label{parameters}
A example for a parameter file is given below:
\begin{verbatim}
&lt;parameter xmlns="http://openmbv.berlios.de/MBXMLUtils/parameter"&gt;
  &lt;vectorParameter name="a"&gt;[1;2;3]*N&lt;/scalarParameter&gt;
  &lt;scalarParameter name="N"&gt;&lt;9/scalarParameter&gt;
  &lt;scalarParameter name="lO"&gt;0.2*N&lt;/scalarParameter&gt;
  &lt;matrixParameter name="A"&gt;[1,2;3,4]&lt;/scalarParameter&gt;
&lt;/parameter&gt;
\end{verbatim}
The parameter names must be unique and the parameters can have references to each other. The order of scalar, vector and matrix parameters is arbitary. The parameter values can be given as \hyperref[octave]{Octave Expressions/Programs}~(P.~\pageref*{octave}).

\chapter{Octave Expression/Program}
\label{octave}
A octave expression/program can be arbitary octave code. So it can be a single statement or a statement list.

If it is a single statement, then the value for the XML element is just the value of the evaluated octave statement. The type of this value must match the type of the XML element (scalar, vector or matrix). The following examples shows valid examples for a single octave statement (one per line), if a scalar \hyperref[parameters]{parameter}~(P.~\pageref*{parameters}) of name \verb|a| and \verb|b| exist:
\begin{verbatim}
4
b
3+a*8
[4;a]*[6,b]
\end{verbatim}

If the text is a statement list, then the value for the XML element is the value of the variable 'ret' which must be set by the statement list. The type of the variable 'ret' must match the type of the XML element (scalar, vector or matrix). The following examples shows valid examples for a octave statement list (one per line), if a scalar \hyperref[parameters]{parameter}~(P.~\pageref*{parameters}) of name \verb|a| and \verb|b| exist:
\begin{verbatim}
if 1==a; ret=4; else ret=8; end
myvar=[1;a];myvar2=myvar*2;ret=myvar2*b;dummy=3
\end{verbatim}

A octave expression can also expand over more then one line like below. Note that the characters '\verb|&amp;|', '\verb|&lt;|' and '\verb|&gt;|' are not allowed in XML. So you have to quote them with '\verb|&amp;amp;|', '\verb|&amp;lt;|' and '\verb|&amp;gt;|', or you must enclose the octave code in a CDATA section:
\begin{verbatim}
&lt;![CDATA[
function r=myfunc(a)
  r=2*a;
end
if 1 &amp; 2, x=9; else x=8; end
ret=myfunc(m1/2);
]]&gt;
\end{verbatim}

\chapter{Embeding}
\label{embed}
Using the \lstinline[basicstyle=\ttfamily]|&lt;pv:embed&gt;| element, where the prefix \lstinline[basicstyle=\ttfamily]|pv| is mapped to the namespace-uri \lstinline[basicstyle=\ttfamily]|http://openmbv.berlios.de/MBXMLUtils/physicalvariable| it is possible to embed a XML element multiple times. The full valid example syntax for this element is:
\begin{verbatim}
&lt;pv:embed href="file.xml" count="2+a" counterName="n" onlyif="n!=2"/&gt;
\end{verbatim}
or
\begin{verbatim}
&lt;pv:embed count="2+a" counterName="n" onlyif="n!=2"&gt;
  &lt;any_element_with_childs/&gt;
&lt;/pv:embed&gt;
\end{verbatim}
This will substitute the \lstinline[basicstyle=\ttfamily]|&lt;pv:embed&gt;| element in the current context \lstinline[basicstyle=\ttfamily]|2+a| times with the element defined in the file \lstinline[basicstyle=\ttfamily]|file.xml| or with \lstinline[basicstyle=\ttfamily]|&lt;any_element_with_childs&gt;|. The insert elements have access to a parameter named \lstinline[basicstyle=\ttfamily]|n| which counts from \lstinline[basicstyle=\ttfamily]|1| to \lstinline[basicstyle=\ttfamily]|2+a| for each insert element. The new element is only insert if the octave expression defined by the attribute \lstinline[basicstyle=\ttfamily]|onlyif| evaluates to \lstinline[basicstyle=\ttfamily]|1| (\lstinline[basicstyle=\ttfamily]|true|). If the attribute \lstinline[basicstyle=\ttfamily]|onlyif| is not given it is allways \lstinline[basicstyle=\ttfamily]|1| (\lstinline[basicstyle=\ttfamily]|true|).\\
The attributes \lstinline[basicstyle=\ttfamily]|count| and \lstinline[basicstyle=\ttfamily]|counterName| must be given both or none of them. If none are given, then \lstinline[basicstyle=\ttfamily]|count| is \lstinline[basicstyle=\ttfamily]|1| and \lstinline[basicstyle=\ttfamily]|counterName| is not used.

The first child element of \lstinline[basicstyle=\ttfamily]|&lt;pv:embed&gt;| can be the element \lstinline[basicstyle=\ttfamily]|&lt;pv:localParameter&gt;| which has one child element \hyperref[parameters]{\lstinline[basicstyle=\ttfamily]|&lt;p:parameter&gt;|}~(P.~\pageref*{parameters}) OR a attribute named \texttt{href}. In this case the global parameters are expanded by the parameters given by the element \lstinline[basicstyle=\ttfamily]|&lt;p:parameter&gt;| or in the file given by \texttt{href}. If a parameter already exist then the parameter is overwritten.
\begin{verbatim}
&lt;pv:embed count="2+a" counterName="n" onlyif="n!=2"&gt;
  &lt;pv:localParameter&gt;
    &lt;p:parameter xmlns:p="http://openmbv.berlios.de/MBXMLUtils/parameter"&gt;
      &lt;p:scalarParameter name="h1"&gt;0.5&lt;/p:scalarParameter&gt;
      &lt;p:scalarParameter name="h2"&gt;h1&lt;/p:scalarParameter&gt;
    &lt;/p:parameter&gt;
  &lt;/pv:localParameter&gt;
  &lt;any_element_with_childs/&gt;
&lt;/pv:embed&gt;
\end{verbatim}

\chapter{Measurement}
\label{measurement}
The following measurements are defined
    <xsl:apply-templates select="/mm:measurement/mm:measure">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

\end{document}
  </xsl:template>

  <xsl:template match="/mm:measurement/mm:measure">
\section{\lstinline[basicstyle=\ttfamily]|<xsl:value-of select="@name"/>|}
\label{<xsl:value-of select="@name"/>}
The SI unit of \lstinline[basicstyle=\ttfamily]|<xsl:value-of select="@name"/>| is: \lstinline[basicstyle=\ttfamily\bfseries]|<xsl:value-of select="@SIunit"/>|

The following units are defined the measure <xsl:value-of select="@name"/>. "Unit Name" is the name of the unit and
"Conversion to SI Unit" is a expression which converts a value of this unit to the SI unit.

\begin{tabular}{|l|l|}
\hline
\textbf{Unit Name} &amp; \textbf{Conversion to SI Unit} \\
\hline
      <xsl:apply-templates select="mm:unit"/>
\hline
\end{tabular}

  </xsl:template>

  <xsl:template match="mm:unit">
    <!-- if SI unit use bold font -->
    <xsl:if test="../@SIunit=@name">
      \lstinline[basicstyle=\ttfamily\bfseries]|<xsl:value-of select="@name"/>| &amp;
    </xsl:if>
    <!-- if not SI unit use normalt font -->
    <xsl:if test="../@SIunit!=@name">
      \verb|<xsl:value-of select="@name"/>| &amp;
    </xsl:if>
    <!-- outout conversion -->
    \verb|<xsl:value-of select="."/>|\\
  </xsl:template>

</xsl:stylesheet>
