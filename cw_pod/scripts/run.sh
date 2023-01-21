#!/bin/bash

rm -rf res png anim.gif
mkdir res png
echo "Running solution"
./"$1" "$3" < default.in
echo "Running conv.py"
for (( k = 0; k < $2; k++ ));
do
    bin="./res/$k.data"
    if test -f "$bin"; then
        i=$(printf "%03d" $((k+1)))
        png="./png/$i.png"
        python3 conv.py "$bin" "$png"
    else
        break
    fi
done
echo "Running ffmpeg"
cd png
# ffmpeg -framerate 15 -i %03d.png ../anim.gif
ffmpeg -framerate 15 -i %03d.png -pix_fmt yuv420p ../vid.mp4
