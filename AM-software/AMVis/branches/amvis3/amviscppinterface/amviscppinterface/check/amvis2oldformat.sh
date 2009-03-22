#! /bin/sh

FILE=$1.amvis

rm -f *.body
rm -f *.pos

xsltproc $(dirname $0)/amvis2oldformat.xsl $FILE.xml | sed -re "s/\[//g;s/]//g;s/;/ /g;/^\{/s/ /\n/g" | sed -re "s/\{//g;s/\}//g" > out.dat

DATA=$(grep "FILENAME:" out.dat | sed -re "s/^FILENAME: \.(.*)$/\1/;s|\.|/|g")
for D in $DATA; do
  POSFILE=$(echo $FILE.h5/$D | sed -re "s|/|\.|g").0001.pos
  echo "Create pos-file: $POSFILE"
  ../../local/bin/h5dumpserie -s $FILE.h5/$D > $POSFILE.2
  MAXCOLS=$(wc -L $POSFILE.2 | cut -d' ' -f1)
  MAXLINE=$(wc -l $POSFILE.2 | cut -d' ' -f1)
  sed -re "s/^(.*)$/\1                                                                                                                                                                                                                                                                                         /;s/^(.{$MAXCOLS}).*$/\1/" $POSFILE.2 > $POSFILE
done

for B in $(sed -rne "/^FILENAME:/=" out.dat); do
  E=$(sed -rne "$[$B+1],$$p" out.dat | sed -rne "/^FILENAME:/=" | head -n 1)
  test "_$E" = "_" && E=$(wc -l out.dat | cut -d' ' -f1)
  sed -rne "$[$B+1],$$p" out.dat | sed -rne "1,$[$E-1]p" > $(sed -rne "$[$B]p" out.dat | sed -re "s/^FILENAME: \.(.*)$/$FILE.h5.\1.body/")
done

rm -r *.pos.2 out.dat
