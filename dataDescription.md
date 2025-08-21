GIDS Biomedical Data Science Hackathon 2025:
Predicting gene expression using perturb-seq data

Overview

Genetic interactions are important for understanding human disease. Genetic interactions occur when the effects of perturbing two genes differ from what you’d predict based on either gene alone. An extreme example is a synthetic lethal pair, where knock out of either gene is viable and knock out of both genes is lethal. Genetic interactions are relevant to many human genetic diseases, particularly cancer where tumors often require combinations of multiple driver mutations in different genes. Genetic interactions are widespread but difficult to identify. A single genetic interaction can be tested by perturbing (knock out or over-expression) two genes separately and comparing this to a double perturbation. However, with ~20k human genes, there are far too many pairwise interactions to test experimentally (~4e+8).

The goal of this hackathon is to predict gene expression when two genes are perturbed (over-expressed). Early work on this problem showed that double (two gene) perturbations could be predicted from single gene perturbations using gene expression data PMC6746554. For example, if two pathways need to be activated in order to get cell division, and activation of each pathway shows different gene expression signature, then one might be able to predict pairs of genes that activate both pathways by their individual gene expression profiles. This has led to significant interest and modeling efforts to predict the effects of single and double perturbations on gene expression.

The data underlying this hackathon come from perturb-seq experiments. Perturb-seq pairs CRISPR-based screens with single-cell RNA sequencing. Each cell gets a different perturbation (over-expression of one or two genes) and the cell’s gene expression profile can be obtain from single-cell RNA-sequencing. Thus, similar to other single cell RNA sequencing (scRNA-seq) experiments, the data is sparse in that the expression of only a subset of genes is measured from each cell.

Submission

Predictions should be posted to your github account as a plain text file with three comma delimited columns with a header (gene, perturbation, expression), the first column should contain the gene name (e.g. g0902), the second column should contain the double perturbation name (e.g. g0160+g0495), the third column should contain the predicted expression level of the gene in the first column (a numeric value). There should be 50k predictions (1000 genes x 50 perturbations). Predictions will be compared to the average observed expression in experimental data.

Your predictions must be in prediction/prediction.csv in a repository with the name Hackathon-Summer-2025 owned by the team captain. Improperly formatted files will not be scored.

An example prediction file is in Rochester-Biomedical-DS/Hackathon-Summer-2025/prediction/prediction.csv. You should replace this file with your own predictions. If you change the name or format of the file, it will not be scored.

Scoring

To submit predictions, please make sure that your properly formatted prediction.csv file is in your forked repository of '/Hackathon-Summer-2025/main/prediction/prediction.csv.' Captains must submit their own predictions. Any use of predictions from other teams is disqualifying.

Prediction files associated with hackathon registered github accounts will be assessed each day of the competition.

Scores based on the smallest RMSD (root mean squared deviation) will be posted at 3PM daily. To receive a score, please post your predictions by 2:59pm.

Prizes

The competition runs through 2:59 PM EDT 23-August-2025. The predictions each team has committed to their repository at that time will be used to determine their final score. Winning teams must submit their code to organizers to claim their prize.

Prizes

First place in each division: $300 + $75 x (team size)
Second place in each division: 0 + $50 x (team size