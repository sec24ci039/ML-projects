# Job Recommendation Model

This project provides a simple machine learning pipeline that recommends a job role based on education level and a few binary skill indicators.

## Training

```
python job_recommender.py train
```

This command uses the sample dataset in `data/sample_job_data.csv` and saves a model under `models/`.

## Prediction

After training, you can make a prediction by passing the profile information:

```
python job_recommender.py predict --education Masters --python 1 --sql 1 --statistics 1 --java 0 --cloud 0
```

The script will output the most suitable job role for the provided skillset and education level.
