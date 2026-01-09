# Arrest Demographics Disparity Analysis Report

## Overview

This analysis examines FBI 2019 arrest data (Table 43A) to identify which crimes
show the highest racial disproportion relative to the US population (2019 ACS Census).

## Methodology

- **Relative Risk Ratio (RRR)**: RRR = (% of Arrestees in Group) / (% of Population in Group)
- RRR = 1.0 indicates parity with population
- RRR > 1.0 indicates over-representation in arrests
- RRR < 1.0 indicates under-representation in arrests

## Top 10 Crimes by Highest Disparity

| Rank | Crime | Most Over-Represented Group | RRR |
|------|-------|----------------------------|-----|
| 1 | Drunkenness | American_Indian | 12.27 |
| 2 | Gambling | Pacific_Islander | 10.41 |
| 3 | Liquor laws | American_Indian | 7.56 |
| 4 | Offenses against the family and children | American_Indian | 7.13 |
| 5 | Disorderly conduct | American_Indian | 6.90 |
| 6 | Suspicion | American_Indian | 5.47 |
| 7 | Vagrancy | American_Indian | 4.35 |
| 8 | Robbery | Black | 4.26 |
| 9 | Murder and nonnegligent manslaughter | Black | 4.14 |
| 10 | Aggravated assault | American_Indian | 3.85 |

## Clustering Analysis

Crimes were clustered into 4 groups using K-Medoids (PAM-approximation).

### Cluster 0: American_Indian-heavy crimes

- **N Crimes**: 16
- **Dominant RRR**: 2.89
- **Crimes**: Rape, Aggravated assault, Burglary, Larceny-theft, Motor vehicle theft, Arson, Property crime, Other assaults, Forgery and counterfeiting, Fraud, Vandalism, Sex offenses (except rape and prostitution), Drug abuse violations, Driving under the influence, All other offenses (except traffic), Curfew and loitering law violations

### Cluster 1: American_Indian-heavy crimes

- **N Crimes**: 6
- **Dominant RRR**: 7.28
- **Crimes**: Offenses against the family and children, Liquor laws, Drunkenness, Disorderly conduct, Vagrancy, Suspicion

### Cluster 2: Pacific_Islander-heavy crimes

- **N Crimes**: 2
- **Dominant RRR**: 6.51
- **Crimes**: Prostitution and commercialized vice, Gambling

### Cluster 3: Black-heavy crimes

- **N Crimes**: 6
- **Dominant RRR**: 3.41
- **Crimes**: Murder and nonnegligent manslaughter, Robbery, Violent crime, Embezzlement, Stolen property; buying, receiving, possessing, Weapons; carrying, possessing, etc.

## Visualizations

See the `outputs/figures/` directory for:

1. **rrr_disparity_heatmap.png**: Log-scale heatmap of RRR by crime × race
2. **crime_dendrogram.png**: Hierarchical clustering taxonomy
3. **cluster_profiles.png**: Mean RRR per cluster
4. **gmm_probabilities.png**: Soft cluster membership probabilities