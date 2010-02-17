#! /bin/sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <h5-infile> <h5-outfile> <startind> <inc>"
  echo ""
  echo "Read each dataset in <h5-infile> and use only rows from <startind>"
  echo "to <endindex> with stepsize <inc> and output to <h5-outfile>"
  echo "NOTE: The index of the first row is 1."
  exit
fi

INFILE=$1
OUTFILE=$2
START=$3
INC=$4

test -d OUT || mkdir OUT
rm -f $OUTFILE
for i in $(~/project/MBSimNeu/local/bin/h5lsserie $INFILE | grep "(Path: " | cut -d':' -f2 | cut -d' ' -f2 | cut -d')' -f1); do
  mkdir -p OUT/$(dirname $i)
  ~/project/MBSimNeu/local/bin/h5dumpserie $i | grep -v "^#" | sed -nre "$START~${INC}p" > OUT/$i
  ROWS=$(cat OUT/$i | wc -l)
  COLS=$(head -n 1 OUT/$i | wc -w)
  CHUNK=1000
  if [ $ROWS -le $CHUNK ]; then CHUNK=$ROWS; fi
  echo PATH $(echo $i | cut -d'/' -f2-) > OUT/$i.config
  echo INPUT-CLASS TEXTFP >> OUT/$i.config
  echo INPUT-SIZE 64 >> OUT/$i.config
  echo RANK 2 >> OUT/$i.config
  echo DIMENSION-SIZES $ROWS $COLS >> OUT/$i.config
  echo CHUNKED-DIMENSION-SIZES $CHUNK $COLS >> OUT/$i.config
  echo MAXIMUM-DIMENSIONS -1 $COLS >> OUT/$i.config
  echo OUTPUT-CLASS FP >> OUT/$i.config
  echo OUTPUT-SIZE 64 >> OUT/$i.config
  h5import OUT/$i -c OUT/$i.config -o $OUTFILE
done
