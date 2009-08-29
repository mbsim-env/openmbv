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
Where \texttt{pv} is mapped to the namespace-uri \texttt{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}.

The content of a scalar type must be a text value. All parameters defined in the parameter file given to the XML preprocessor are substututed in this text value. Afterwards this text value, which can be a abitary octave  string/program, is evaluated by octave. The text value must evalute to a single scalar value. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
\begin{verbatim}
  &lt;myScalarElement&gt;4*sin(a)+b&lt;/myScalarElement&gt;
  &lt;myScalarElement&gt;[a,2]*[3;b]&lt;/myScalarElement&gt;
\end{verbatim}

\chapter{Vector Type}
\label{vectortype}
A vector type can be of any unit defined in \hyperref[measurement]{measurements}.
The type name of a vector of measure length is \texttt{pv:lengthVector} and so on.
Where \texttt{pv} is mapped to the namespace-uri \texttt{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}.

The content of a vector type can be one of the following:
\begin{itemize}
  \item A text value: All parameters defined in the parameter file given to the XML preprocessor are substututed in this text value. Afterwards this text value, which can be a abitary octave  string/program, is evaluated by octave. The text value must evalute to a single row vector value. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
    \begin{verbatim}
&lt;myVectorElement&gt;[1;b;a;7]&lt;/myVectorElement&gt;
&lt;myVectorElement&gt;[a,2;5.6,7]*[3;b]&lt;/myVectorElement&gt;
  \item A XML representation of a vector: The following shows a example of such a XML representation.
    \begin{verbatim}
&lt;myVectorElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;7.3&lt;/pv:ele&gt;
&lt;/myVectorElement&gt;
    \end{verbatim}
  \item A reference to a ascii file: The following shows a example as a reference to the file vec.txt.
    \begin{verbatim}
&lt;myVectorElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:asciiVectorRef href="vec.txt"/&gt;
&lt;/myVectorElement&gt;
    \end{verbatim}
The file vec.txt is a simple ascii file containing one element of the vector per line. All empty lines are ignored and the the content between '\#' or '\%' and the end of line is also ignored (comments).
\end{itemize}

\chapter{Matrix Type}
\label{matrixtype}
A matrix type can be of any unit defined in \hyperref[measurement]{measurements}.
The type name of a matrix of measure length is \texttt{pv:lengthMatrix} and so on.
Where \texttt{pv} is mapped to the namespace-uri \texttt{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}.

The content of a matrix type can be one of the following:
\begin{itemize}
  \item A text value: All parameters defined in the parameter file given to the XML preprocessor are substututed in this text value. Afterwards this text value, which can be a abitary octave  string/program, is evaluated by octave. The text value must evalute to a single matrix value. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
    \begin{verbatim}
&lt;myMatrixElement&gt;[1,b;a,7]&lt;/myMatrixElement&gt;
&lt;myMatrixElement&gt;[a,2;5.6,7]*rand(2,2)&lt;/myMatrixElement&gt;
    \end{verbatim}
  \item A XML representation of a matrix: The following shows a example of such a XML representation.
    \begin{verbatim}
&lt;myMatrixElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
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
&lt;myMatrixElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:asciiMatrixRef href="mat.txt"/&gt;
&lt;/myMatrixElement&gt;
    \end{verbatim}
    The file mat.txt is a simple ascii file containing one row of the vector per line. The values inside a row must be separated by ',' or space. All empty lines are ignored and the the content between '\#' or '\%' and the end of line is also ignored (comments).
\end{itemize}

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
