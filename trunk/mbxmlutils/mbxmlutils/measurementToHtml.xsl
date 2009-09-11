<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:mm="http://openmbv.berlios.de/MBXMLUtils/measurement"
  xmlns="http://www.w3.org/1999/xhtml"
  version="1.0">

  <!-- If changes in this file are made, then the analog changes must
       be done in the file measurementToTex.xsl -->

  <!-- output method -->
  <xsl:output method="xml"
    encoding="UTF-8"
    doctype-public="-//W3C//DTD XHTML 1.0 Strict//EN"
    doctype-system="http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <!-- for all direct root child elements (classes) -->
  <xsl:template match="/">
    <!-- html header -->
    <html xml:lang="en" lang="en"><head><title>Measurement - XML Documentation</title></head><body>
    <h1>Measurement - XML Documentation</h1>
    <h2>Contents</h2>
    <ul>
      <li><a name="content-introduction" href="#introduction">Introduction</a></li>
      <li><a name="content-scalartype" href="#scalartype">Scalar Type</a></li>
      <li><a name="content-vectortype" href="#vectortype">Vector Type</a></li>
      <li><a name="content-matrixtype" href="#matrixtype">Matrix Type</a></li>
      <li><a name="content-octave" href="#octave">Octave Expression/Program</a></li>
      <li><a name="content-embed" href="#embed">Embeding</a></li>
      <li><a name="content-measurements" href="#measurements">Measurements</a>
        <ul>
          <xsl:for-each select="/mm:measurement/mm:measure">
            <xsl:sort select="@name"/>
            <li>
              <a><xsl:attribute name="name">content-<xsl:value-of select="@name"/></xsl:attribute>
                <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute><xsl:value-of select="@name"/></a>
            </li>
          </xsl:for-each>
        </ul>
      </li>
    </ul>

    <h2><a name="introduction" href="#content-introduction">Introduction</a></h2>
    <p>This is the documentation of the definition of scalar, vector and matrix types with a abitary measurement.</p>

    <h2><a name="scalartype" href="#content-scalartype">Scalar Type</a></h2>
    <p>A scalar type can be of any unit defined in <a href="#measurements">measurements</a>.
      The type name of a scalar of measure length is <span style="font-family:monospace">pv:lengthScalar</span> and so on.
      Where <span style="font-family:monospace">pv</span> is mapped to the namespace-uri <span style="font-family:monospace">http://openmbv.berlios.de/MBXMLUtils/physicalvariable</span>.</p>
    <p>The content of a scalar type must be a <a href="#octave">octave expression/program</a>. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:</p>
    <pre>&lt;myScalarElement&gt;4*sin(a)+b&lt;/myScalarElement&gt;</pre>
    <pre>&lt;myScalarElement&gt;[a,2]*[3;b]&lt;/myScalarElement&gt;</pre>

    <h2><a name="vectortype" href="#content-vectortype">Vector Type</a></h2>
    <p>A vector type can be of any unit defined in <a href="#measurements">measurements</a>.
      The type name of a vector of measure length is <span style="font-family:monospace">pv:lengthVector</span> and so on.
      Where <span style="font-family:monospace">pv</span> is mapped to the namespace-uri <span style="font-family:monospace">http://openmbv.berlios.de/MBXMLUtils/physicalvariable</span>.</p>
    <p>The content of a vector type can be one of the following:</p>
    <ul>
      <li>A <a href="#octave">octave expression/program</a>. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
          <pre>&lt;myVectorElement&gt;[1;b;a;7]&lt;/myVectorElement&gt;</pre>
          <pre>&lt;myVectorElement&gt;[a,2;5.6,7]*[3;b]&lt;/myVectorElement&gt;</pre></li>
      <li>A XML representation of a vector: The following shows a example of such a XML representation.<pre>
&lt;myVectorElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;7.3&lt;/pv:ele&gt;
&lt;/myVectorElement&gt;
</pre></li>
      <li>A reference to a ascii file: The following shows a example as a reference to the file vec.txt.<pre>
&lt;myVectorElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:asciiVectorRef href="vec.txt"/&gt;
&lt;/myVectorElement&gt;
</pre>The file vec.txt is a simple ascii file containing one element of the vector per line. All empty lines are ignored and the the content between '#' or '%' and the end of line is also ignored (comments).</li>
    </ul>

    <h2><a name="matrixtype" href="#content-matrixtype">Matrix Type</a></h2>
    <p>A matrix type can be of any unit defined in <a href="#measurements">measurements</a>.
      The type name of a matrix of measure length is <span style="font-family:monospace">pv:lengthMatrix</span> and so on.
      Where <span style="font-family:monospace">pv</span> is mapped to the namespace-uri <span style="font-family:monospace">http://openmbv.berlios.de/MBXMLUtils/physicalvariable</span>.</p>
    <p>The content of a matrix type can be one of the following:</p>
    <ul>
      <li>A <a href="#octave">octave expression/program</a>. The following examples are valid, if there exist a scalar paremter a and b in the parameter file:
          <pre>&lt;myMatrixElement&gt;[1,b;a,7]&lt;/myMatrixElement&gt;</pre>
          <pre>&lt;myMatrixElement&gt;[a,2;5.6,7]*rand(2,2)&lt;/myMatrixElement&gt;</pre></li>
      <li>A XML representation of a matrix: The following shows a example of such a XML representation.<pre>
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
</pre></li>
      <li>A reference to a ascii file: The following shows a example as a reference to the file mat.txt.<pre>
&lt;myMatrixElement xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"&gt;
  &lt;pv:asciiMatrixRef href="mat.txt"/&gt;
&lt;/myMatrixElement&gt;
</pre>The file mat.txt is a simple ascii file containing one row of the vector per line. The values inside a row must be separated by ',' or space. All empty lines are ignored and the the content between '#' or '%' and the end of line is also ignored (comments).</li>
    </ul>

    <h2><a name="octave" href="#content-octave">Octave Expression/Program</a></h2>
    <p>A octave expression/program can be arbitary octave code. So it can be a single statement or a statement list.</p>

   <p>If it is a single statement, then the value for the XML element is just the value of the evaluated octave statement. The type of this value must match the type of the XML element (scalar, vector or matrix). The following examples shows valid examples for a single octave statement (one per line), if a scalar parameter of name 'a' and 'b' exist:</p>
<pre>4
b
3+a*8
[4;a]*[6,b]
</pre>

<p>If the text is a statement list, then the value for the XML element is the value of the variable 'ret' which must be set by the statement list. The type of the variable 'ret' must match the type of the XML element (scalar, vector or matrix). The following examples shows valid examples for a octave statement list (one per line), if a scalar parameter of name 'a' and 'b' exist:</p>
<pre>if 1==a; ret=4; else ret=8; end
myvar=[1;a];myvar2=myvar*2;ret=myvar2*b;dummy=3
</pre>

<p>A octave expression can also expand over more then one line like below. Note that the characters '&amp;', '&lt;' and '&gt;' are not allowed in XML. So you have to quote them with '&amp;amp;', '&amp;lt;' and '&amp;gt;', or you must enclose the octave code in a CDATA section:</p>
<pre>&lt;![CDATA[
function r=myfunc(a)
  r=2*a;
end
if 1 &amp; 2, x=9; else x=8; end
ret=myfunc(m1/2);
]]&gt;
</pre>

    <h2><a name="embed" href="#content-embed">Embeding</a></h2>
    <p>Using the <span style="font-family:monospace">&lt;pv:embed&gt;</span> element, where the prefix <span style="font-family:monospace">pv</span> is mapped to the namespace-uri <span style="font-family:monospace">http://openmbv.berlios.de/MBXMLUtils/physicalvariable</span> it is possible to embed a XML element multiple times. The full valid example syntax for this element is:</p>
<pre>&lt;pv:embed href="file.xml" count="2+a" counterName="n" onlyif="n!=2"/&gt;</pre>
<p>or</p>
<pre>&lt;pv:embed count="2+a" counterName="n" onlyif="n!=2"&gt;
  &lt;any_element_with_childs/&gt;
&lt;/pv:embed&gt;
</pre>
<p>This will substitute the <span style="font-family:monospace">&lt;pv:embed&gt;</span> element in the current context <span style="font-family:monospace">2+a</span> times with the element defined in the file <span style="font-family:monospace">file.xml</span> or with <span style="font-family:monospace">&lt;any_element_with_childs&gt;</span>. The insert elements have access to the global parameters and to a parameter named <span style="font-family:monospace">n</span> which counts from <span style="font-family:monospace">1</span> to <span style="font-family:monospace">2+a</span> for each insert element. The new element is only insert if the octave expression defined by the attribute <span style="font-family:monospace">onlyif</span> evaluates to <span style="font-family:monospace">1</span> (<span style="font-family:monospace">true</span>). If the attribute <span style="font-family:monospace">onlyif</span> is not given it is allways <span style="font-family:monospace">1</span> (<span style="font-family:monospace">true</span>).</p>

<p>The first child element of <span style="font-family:monospace">&lt;pv:embed&gt;</span> can be the element <span style="font-family:monospace">&lt;p:parameter&gt;</span>. In this case the global parameters are expanded by the parameters given by this element. If a parameter already exist then the parameter is overwritten.</p>
<pre>&lt;pv:embed count="2+a" counterName="n" onlyif="n!=2"&gt;
  &lt;p:parameter xmlns:p="http://openmbv.berlios.de/MBXMLUtils/parameter"&gt;
    &lt;p:scalarParameter name="h1"&gt;0.5&lt;/p:scalarParameter&gt;
    &lt;p:scalarParameter name="h2"&gt;h1&lt;/p:scalarParameter&gt;
  &lt;/p:parameter&gt;
  &lt;any_element_with_childs/&gt;
&lt;/pv:embed&gt;
</pre>

    <h2><a name="measurements" href="#content-measurements">Measurements</a></h2>
    <p>The following measurements are defined</p>
    <xsl:apply-templates select="/mm:measurement/mm:measure">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

    </body></html>
  </xsl:template>

  <xsl:template match="/mm:measurement/mm:measure">
    <h3><a>
      <xsl:attribute name="name">
        <xsl:value-of select="@name"/>
      </xsl:attribute>
      <xsl:attribute name="href">#content-<xsl:value-of select="@name"/></xsl:attribute>
      <xsl:value-of select="@name"/>
    </a></h3>
    <p>The SI unit of <xsl:value-of select="@name"/> is: <span style="font-weight:bold"><xsl:value-of select="@SIunit"/></span></p>
    <p>The following units are defined the measure <xsl:value-of select="@name"/>. "Unit Name" is the name of the unit and
      "Conversion to SI Unit" is a expression which converts a value of this unit to the SI unit.</p>
    <table border="1">
      <thead> <tr> <th>Unit Name</th> <th>Conversion to SI Unit</th> </tr> </thead>
      <xsl:apply-templates select="mm:unit"/>
    </table>
  </xsl:template>

  <xsl:template match="mm:unit">
    <tr>
      <!-- if SI unit use bold font -->
      <xsl:if test="../@SIunit=@name">
        <td><span style="font-weight:bold"><xsl:value-of select="@name"/></span></td>
      </xsl:if>
      <!-- if not SI unit use normalt font -->
      <xsl:if test="../@SIunit!=@name">
        <td><xsl:value-of select="@name"/></td>
      </xsl:if>
      <!-- outout conversion -->
      <td><xsl:value-of select="."/></td>
    </tr>
  </xsl:template>

</xsl:stylesheet>