# relatedness
We provide the implementation for [Learning Relatedness between Types with Prototypes for Relation Extraction](https://www.aclweb.org/anthology/2021.eacl-main.172/)
# Environment
python==2.7
tensorflow==1.4
# Dataset
ACE: ACE2005 (https://catalog.ldc.upenn.edu/LDC2006T06)

ERE: LDC2015E29, LDC2015E68, LDC2015E78, LDC2015R26, LDC2016E31, LDC2016E73

Both datasets are available through LDC.

We processed the LDC data and transformed them into intermediate files.
An example of the format is located at `data/ERE/ere_directional_example.txt`.
These are the input files to our implementation.

# Implementation
The model implementation is available in this code repository. It also contains redundant codes that were used for various experiments, but not necessarily for this paper.

An example command:
`python multi.py --expt=ace05_ere --joint=joint_ksupport --debug`

Note: the input processed data files are not provided in the repository.