#! /bin/sh

V='([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)' # a floating number value with or wihtout exponent and without a signum
#V='(([0-9]+|\.[0-9]+|[0-9]+\.|[0-9]+\.[0-9]+)([eE][-+]?[0-9]+)?)' # a floating number value with or wihtout exponent and without a signum
V='(([0-9]+|\.[0-9]+|[0-9]+\.|[0-9]+\.[0-9]+)([eE][-+]?[0-9]+)?)' # a floating number value with or wihtout exponent and without a signum
P='([a-zA-Z_][a-zA-Z_0-9]*)' # a parameter
O='[-+*/^]' # a bi-nary arithmetic operator
S='[-+]' # a signum

N='(V|P)' # a number (vlue or parameter)
N=$(echo $N | sed -re "s@V@$V@g;s@P@$P@g;")
E='S?N(ON)*' # a expression (sequence of number and operator)
E=$(echo $E | sed -re "s@S@$S@g;s@N@$N@g;s@O@$O@g;")
F='((sin|cos|tan|)\(E\))' # a function of a expression or just a bracket
F=$(echo $F | sed -re "s@E@$E@g;")


EQU='5*sin(abc+cos(5.1+7)*6.5/3.4e-6)+5.6'

while [[ $EQU =~ '(' ]]; do 
  echo $EQU
  EQU=$(echo $EQU | sed -re "s/$F/dummy/")
done
echo $EQU
EQU=$(echo $EQU | sed -re "s/$E//")

if [ -z $EQU ]; then
  echo true
else
  echo false
fi
#
## recursion!!!
#N = (V|P|F)
#
#
#
#
#
#
## Vector (column)
#V = \[ *F( *; *F)* *\]
#
## Matrix
#M = \[ *F( *[, ] *F)* *(; *F( *[, ] *F)*) *\]
