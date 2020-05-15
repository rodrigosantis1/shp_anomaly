# Anomaly detection in SHPs
Intelligent fault diagnosis based on Isolation Forest algorithm for Small Hydroelectric Plants (SHPs).

## Requirements
The following softwares/packages are required for running the scripts:
- Python 3.7.6
- SciPy 1.4.1
- Scikit-learn 0.22.1
- Pandas 1.0.1
- NumPy 1.18.1
- Matplotlib 3.1.3

## Files
data
    /faults.csv - all registered faults during the comprised period;
    /test.csv - contains unhealthy/abnormal operation state. Includes 12h operation data prior failure;
    /train.csv - contains healthy operation state.
isolation_forest.py - contains script for simulating isolation forest model, PCA and KICA-PCA methods.
results.csv - results file generated from isolation_forest.py execution.
