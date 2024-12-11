# LLM-for-data-cleaning
CS386D term project
## Overview
This project fouses on utilizing LLM for data cleaning based on Movies dataset，you can find the original website [here](https://www.kaggle.com/datasets/bharatnatrayn/movies-dataset-for-feature-extracion-prediction/data). It used Openai GPT-4o-mini to help correct errors introduced to the dataset. The Movies dataset includes attributes such as Movie_name, publish_year, genre, rating etc. You can introduce different rate of errors to the dataset to fully test GPT-4o-mini. While the dataset has around 10k rows, you can modify it but here I truncate them and use about 2k rows for efficiency.

## Dependencies
Python 3.8+
openai 1.5.72
pandas 2.0.3
tqdm 4.66.1

## File description
**add_noise.py** 

Truncate and generate baseline dataset and introduce errors.

**clean_data.py**   

Correct errors we introduced using GPT-4o-mini and evaluate rhe result.

**movies.csv**  

raw dataset

## How to run the project
1. Install dependencies
2. Input yout OPENAI_API_KEY and get access to openai model
3. Run add_noise.py
4. Run clean_data.py
5. Metrics result will show up on the terminal and cleaned data files will be saved in main directory.
