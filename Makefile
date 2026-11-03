.PHONY: test experiments figures demo report all
test:        ; python -m pytest -q
experiments: ; OMP_NUM_THREADS=1 python experiments/run.py diagnose reg & OMP_NUM_THREADS=1 python experiments/run.py main denoise grid control; wait
figures:     ; python experiments/make_figures.py
demo:        ; python experiments/export_demo.py
report:      ; python build_report.py
all: test experiments figures demo report
