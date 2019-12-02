# LREC_TCERiDSI
### This repository represents source code used to get results of paper titled "Complementary Dynamic Time-Continuous Emotion Recognition in Dyadic Spoken Interactions" submitted to LREC 2020 by Dmitrii Fedotov, Heysem Kaya, Alexey Karpov and Wolfgang Minker

Presented pipeline includes pre-processing, modeling and post-processing of data. In order to reproduce results, one should acquire databases from their holders in accordance with EULAs, clean audio files using turns annotation of this data as described in paper, and extract features using openSMILE software (https://www.audeering.com/opensmile/). Links to databases:
1. SEWA: https://db.sewaproject.eu/ (we have used publically available part of the database presented as CES sub-challenge of AVEC2019: https://sites.google.com/view/avec2019)
2. CreativeIT: https://sail.usc.edu/CreativeIT/ImprovRelease.htm
3. UUDB: http://uudb.speech-lab.org/
4. IEMOCAP: https://sail.usc.edu/iemocap/
5. SEMAINE: https://semaine-db.eu/

Note that for SEMAINE database sessions 131-140 of recordings 25 and 26 do not have turns annotations, therefore they were done manually on our side. We have added them to "additional/" folder to save time and effort of future researchers.

Use python script run_CV.py to run n-folded cross validation on prepared data. The script covers complete pipeline, including preprocessing, modelling, prediction and evaluation. See comments in file for more details.
