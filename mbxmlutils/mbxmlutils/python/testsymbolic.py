import mbxmlutils
import sympy

c=3.1
i=sympy.Dummy()
s=2*i

y=c+i+i*s
print(mbxmlutils._serializeFunction(y))

y=sympy.atan2(c, s)
print(mbxmlutils._serializeFunction(y))

y=sympy.Heaviside(s)
print(mbxmlutils._serializeFunction(y))

y=sympy.Min(c, s, i*i)
print(mbxmlutils._serializeFunction(y))

y=sympy.Max(c, s, i*i)
print(mbxmlutils._serializeFunction(y))

y=sympy.Piecewise((s, i>1), (c*c, s<2), (s*i, 2<i))
print(mbxmlutils._serializeFunction(y))
