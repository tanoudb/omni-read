set -u
cd "A:/omni read"
SERIE="path-of-vengeance"
for n in 01 02 03 04; do
  IMG="manhwa/$SERIE/Chapitre 001/Chapitre 001_merged_part$n.jpg"
  OUT="scratch/render_out/S1_LOCK_$n"
  echo "##### DEBUT part$n  $(date +%H:%M:%S) #####"
  PYTHONIOENCODING=utf-8 python scratch/render_iterate.py "$IMG" "$OUT" 2>&1 \
    | grep -viE "futurewarning|@torch|deprecat|warnings.warn" | tail -12
  echo "##### FIN part$n  code=$?  $(date +%H:%M:%S) #####"
done
echo "##### TOUT TERMINE #####"
