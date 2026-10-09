#!/usr/bin/env bash
# Frame code robustness: re-encode a recorded camera at falling bitrates and smaller sizes, count readable pictures.
# Usage: docs/robustness.sh <recording of one 1280x720 camera> [camcheck binary] > docs/robustness.csv
# Record the source first, e.g. ffmpeg -rtsp_transport tcp -i rtsp://127.0.0.1:8554/cam01 -t 12 -c copy src.mkv
# Needs ffmpeg with libx264 and libx265. Then: python3 docs/robustness_chart.py
set -eu
src=$1
camcheck=${2:-camcheck}
work=$(mktemp -d)
trap 'rm -f "$work"/v.* "$work"/r.csv; rmdir "$work"' EXIT
echo "codec,width,height,setting,kbps,frames,readable,correct_camera"
for size in 1280x720 960x540 640x360 480x270 320x180; do
    w=${size%x*}; h=${size#*x}
    for codec in h264 h265 mjpeg; do
        if [ $codec = mjpeg ]; then settings="2 8 16 24 31"; else settings="1000 500 250 125 60 30"; fi
        for s in $settings; do
            case $codec in
                h264) enc="-c:v libx264 -preset medium -b:v ${s}k -maxrate ${s}k -bufsize ${s}k -g 30 -bf 0"; ext=mp4 ;;
                h265) enc="-c:v libx265 -preset medium -b:v ${s}k -maxrate ${s}k -bufsize ${s}k -x265-params keyint=30:bframes=0:log-level=error"; ext=mp4 ;;
                mjpeg) enc="-c:v mjpeg -q:v $s -pix_fmt yuvj420p"; ext=avi ;;
            esac
            out=$work/v.$ext
            ffmpeg -loglevel error -y -i "$src" -vf scale=$w:$h:flags=bicubic $enc -an "$out"
            secs=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")
            kbps=$(python3 -c "import os; print(round(os.path.getsize('$out') * 8 / $secs / 1000))")
            "$camcheck" grid --file "$out" --count 1 --cols 1 --cell-w $w --cell-h $h --warmup 0 --csv "$work/r.csv" > /dev/null 2>&1
            python3 -c "
import csv
rows = list(csv.DictReader(open('$work/r.csv')))
ok = [r for r in rows if not r['error']]
print(f'$codec,$w,$h,$s,$kbps,{len(rows)},{len(ok)},{sum(1 for r in ok if r[\"frame_camera\"] == \"1\")}')"
            rm -f "$out"
        done
    done
done
