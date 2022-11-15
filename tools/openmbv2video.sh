#! /bin/bash

set -e
set -o pipefail

if test $# -ne 2; then
  echo "Usage: openmbv2video <fps> <fileBaseName>"
  echo ""
  echo "<fps>:          Frames per second as defined by the export in OpenMBV"
  echo "<fileBaseName>: The base name of the png files. If there exists files named"
  echo "                'test_000000.png' use simply 'test'."
  echo ""
  echo "This script needs mencoder (mplayer) from http://www.mplayerhq.hu"
  echo "The generated video should work with MS Powerpoint without the installation"
  echo "of additional codecs."
  exit
fi

mencoder mf://$2_*.png -mf fps=$1:type=png -nosound -ovc lavc -lavcopts vcodec=wmv1:vbitrate=5120 -of lavf -o video.wmv
