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

\title{Measurement - XML Documentation}
\maketitle

\tableofcontents

\chapter{Introduction}
\label{introduction}
This is the documentation of the definition of scalar, vector and matrix types with a abitary measurement.

\chapter{Scalar Type}
\label{scalartype}
A scalar type can be of any unit defined in \hyperref[measurement]{measurements}.
The type name of a scalar of measure length is \texttt{pv:lengthScalar} and so on.
Where \texttt{pv} is mapped to the namespace-uri\\
\texttt{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}.

The content of a scalar type must be a \hyperref[octave]{octave expression/program}. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
\begin{verbatim}
  &lt;myScalarElement&gt;4*sin(a)+b&lt;/myScalarElement&gt;
  &lt;myScalarElement&gt;[a,2]*[3;b]&lt;/myScalarElement&gt;
\end{verbatim}

\chapter{Vector Type}
\label{vectortype}
A vector type can be of any unit defined in \hyperref[measurement]{measurements}.
The type name of a vector of measure length is \texttt{pv:lengthVector} and so on.
Where \texttt{pv} is mapped to the namespace-uri\\
\texttt{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}.

The content of a vector type can be one of the following:
\begin{itemize}
  \item A \hyperref[octave]{octave expression/program}. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
    \begin{verbatim}
&lt;myVectorElement&gt;[1;b;a;7]&lt;/myVectorElement&gt;
&lt;myVectorElement&gt;[a,2;5.6,7]*[3;b]&lt;/myVectorElement&gt;
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
  \item A reference to a ascii file: The following shows a example as a reference to the file vec.txt.
    \begin{verbatim}
&lt;myVectorElement
  xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:asciiVectorRef href="vec.txt"/&gt;
&lt;/myVectorElement&gt;
    \end{verbatim}
The file vec.txt is a simple ascii file containing one element of the vector per line. All empty lines are ignored and the the content between '\#' or '\%' and the end of line is also ignored (comments).
\end{itemize}

\chapter{Matrix Type}
\label{matrixtype}
A matrix type can be of any unit defined in \hyperref[measurement]{measurements}.
The type name of a matrix of measure length is \texttt{pv:lengthMatrix} and so on.
Where \texttt{pv} is mapped to the namespace-uri\\
\texttt{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}.

The content of a matrix type can be one of the following:
\begin{itemize}
  \item A \hyperref[octave]{octave expression/program}. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
    \begin{verbatim}
&lt;myMatrixElement&gt;[1,b;a,7]&lt;/myMatrixElement&gt;
&lt;myMatrixElement&gt;[a,2;5.6,7]*rand(2,2)&lt;/myMatrixElement&gt;
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
  \item A reference to a ascii file: The following shows a example as a reference to the file mat.txt.
    \begin{verbatim}
&lt;myMatrixElement
  xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:asciiMatrixRef href="mat.txt"/&gt;
&lt;/myMatrixElement&gt;
    \end{verbatim}
    The file mat.txt is a simple ascii file containing one row of the vector per line. The values inside a row must be separated by ',' or space. All empty lines are ignored and the the content between '\#' or '\%' and the end of line is also ignored (comments).
\end{itemize}

\chapter{Octave Expression/Program}
\label{octave}
A octave expression/program can be arbitary octave code. So it can be a single statement or a statement list.

If it is a single statement, then the value for the XML element is just the value of the evaluated octave statement. The type of this value must match the type of the XML element (scalar, vector or matrix). The following examples shows valid examples for a single octave statement (one per line), if a scalar parameter of name 'a' and 'b' exist:
\begin{verbatim}
4
b
3+a*8
[4;a]*[6,b]
\end{verbatim}

If the text is a statement list, then the value for the XML element is the value of the variable 'ret' which must be set by the statement list. The type of the variable 'ret' must match the type of the XML element (scalar, vector or matrix). The following examples shows valid examples for a octave statement list (one per line), if a scalar parameter of name 'a' and 'b' exist:
\begin{verbatim}
if 1==a; ret=4; else ret=8; end
myvar=[1;a];myvar2=myvar*2;ret=myvar2*b;dummy=3
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
This will substitute the \lstinline[basicstyle=\ttfamily]|&lt;pv:embed&gt;| element in the current context \lstinline[basicstyle=\ttfamily]|2+a| times with the element defined in the file \lstinline[basicstyle=\ttfamily]|file.xml| or with \lstinline[basicstyle=\ttfamily]|&lt;any_element_with_childs&gt;|. The insert elements have access to a parameter named \lstinline[basicstyle=\ttfamily]|n| which counts from \lstinline[basicstyle=\ttfamily]|1| to \lstinline[basicstyle=\ttfamily]|2+a| for each insert element. The new element is only insert if the octave expression defined by the attribute \lstinline[basicstyle=\ttfamily]|onlyif| evaluates to \lstinline[basicstyle=\ttfamily]|1| (\lstinline[basicstyle=\ttfamily]|true|). If the attribute \lstinline[basicstyle=\ttfamily]|onlyif| is not given it is allways \lstinline[basicstyle=\ttfamily]|1| (\lstinline[basicstyle=\ttfamily]|true|).

\chapter{Measurement}
\label{measurement}
The following measurements are defined
    <xsl:apply-templates select="/mm:measurement/mm:measure">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

\end{document}
  </xsl:template>

  <xsl:template match="/mm:measurement/mm:measure">
\section{<xsl:value-of select="@name"/>}
\label{<xsl:value-of select="@name"/>}
The SI unit of <xsl:value-of select="@name"/> is: \lstinline[basicstyle=\ttfamily\bfseries]|<xsl:value-of select="@SIunit"/>|

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
