TXT file description

Each line follows the format:
sample_id label

Labels:
0 = GBM
1 = SBM

Files:
train.txt   : All training samples from both classes.
train_0.txt : GBM training samples only (label 0); used to initialize and update the GBM reference bank.
train_1.txt : SBM training samples only (label 1); used to initialize and update the SBM reference bank.
test.txt    : All test samples from both classes, used for final evaluation.

Example:
case_0001 0
case_0002 1