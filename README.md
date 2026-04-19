# Forecasting Under Regime Instability: Labor Market Signals After COVID-19

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The purpose of this project is to determine how well labor market signals predict future inflation and wage growth after COVID-19, and test whether machine learning models can adapt to regime instability better than traditional approaches.

## Project Organization

```
├── LICENSE          
├── Makefile       
├── README.md         
│
├── data
│   ├── external    
│   ├── interim        
│   ├── processed      <- Final model-ready datasets (wage, price, interaction panels)
│   └── raw            <- Raw FRED pulls and source files
│
├── docs            
│
├── models            
│
├── notebooks
│   ├── 01-eda.ipynb                   
│   ├── 02-feature-engineering.ipynb    
│   ├── 03-baseline-models.ipynb      
│   ├── 04-structural-analysis.ipynb  
│   ├── 05-forecast-benchmarks.ipynb     
│   └── 06-results-and-audit.ipynb       
│
├── pyproject.toml     
│
├── references   
│
├── reports       
│   └── figures     
│
├── requirements.txt   
│
├── setup.cfg      
│
└── regime_instability <- src for the project
    │
    ├── __init__.py       
    │
    ├── config.py              
    │
    ├── data.py                 <- Data loaders (FRED, file ingestion, API handling)
    │
    ├── preprocessing.py     
    │
    ├── features.py           
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── train.py            <- Model training (OLS, Ridge, RF, etc.)
    │   ├── predict.py          <- Forecast generation and evaluation
    │   └── evaluation.py       <- RMSE comparison, horizon analysis, regime splits
    │
    └── visualization.py      
```
