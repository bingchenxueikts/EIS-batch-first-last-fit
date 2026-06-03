Use this when EC-Lab ZFit refuses txt/mpt files.

1. Install packages once:
   py -m pip install numpy scipy

2. Put batch_fit_first_last_EIS.py in the folder with EC-Lab ASCII exports (*.txt or *.mpt).

3. Run:
   py batch_fit_first_last_EIS.py

4. Outputs:
   python_EIS_fit_results/fit_summary.csv
   python_EIS_fit_results/fitted_curves/*.csv

Circuit:
   R1 + (R2 || C2) + (R3 || C3)

The script does not trust EC-Lab cycle number.
It detects true spectra by frequency reset, then fits only FIRST and LAST spectrum.
