This README.md file was generated on 2021-08-23 by Péter Gargya

GENERAL INFORMATION

1. Title of Dataset: Dataset used in the article:  
    "Histological Grade of Endometrioid Endometrial Cancer and Relapse Risk Can Be Predicted With Machine Learning From Gene Expression Data" 

2. Author Information  
A. Principal Investigator Contact Information  
Name: dr. Bálint László Bálint  
Institution: Genomic Medicine and Bioinformatics Core Facility, Department of Biochemistry and Molecular Biology, Faculty of Medicine, University of Debrecen  
Address: Egyetem tér 1, 4032 Debrecen, Hungary  
Email: lbalint@med.unideb.hu  
B. Associate or Co-investigator Contact Information  
Name: Péter Gargya  
Institution: Genomic Medicine and Bioinformatics Core Facility, Department of Biochemistry and Molecular Biology, Faculty of Medicine,   University of Debrecen  
Address: Egyetem tér 1, 4032 Debrecen, Hungary  
Email: gargya.peter@gmail.com  


SHARING/ACCESS INFORMATION  

1. Licenses/restrictions placed on the data: The data and codes provided are free to use, however we kindly ask everybody to cite our article as written below.  

2. Links to publications that cite or use the data:   
https://www.mdpi.com/2072-6694/13/17/4348/htm  

3. Recommended citation for this dataset:   
Gargya, P.; Bálint, B.L. Histological Grade of Endometrioid Endometrial Cancer and Relapse Risk Can Be Predicted with Machine Learning from Gene Expression Data. Cancers 2021, 13, 4348. https://doi.org/10.3390/cancers13174348 

DATA & FILE OVERVIEW  

1. File List:  
- part1_R_prepare_data.Rmd: Codes used to produce the data before applying machine learning.
- part2_Python_ML.py: Creating our Machine Learning model
- part3_R_survival_analysis.Rmd: Survival analysis between low-risk and high-risk G2 subgroups, which were defined by our model.
- part4_extra_requests_by_reviewers.Rmd: Codes used to analyse the distribution of TCGA subgroups inside our risk-specific subgroups.
- ucec_tcga_clinical_data.zip: Raw clinical data, downloaded from cBioportal.
- uterus_rnaseq_VST1.z01, uterus_rnaseq_VST1.z02, uterus_rnaseq_VST1.zip, uterus_rnaseq_VST_G2.zip: These zip files contain the output of       part1_R_prepare_data.Rmd and the input of part2_Python_ML.py

