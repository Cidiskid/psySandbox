prf_path=prf_log/`date | sed "s/ /_/g"`.log
mkdir -p prf_log 2> /dev/null
python -m cProfile -s cumulative mul_control.py > $prf_path