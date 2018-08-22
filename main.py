from SparseDataset import SparseDataset
from MatrixFactorization import MatrixFactorization
import metrics
import random
import pandas as pd


file_path = './data/ratings.csv'

data = SparseDataset(file_path)

# set n_epochs higher for better prediction
algo = MatrixFactorization(n_epochs=2)

# train model
trainset = data.build_trainset()
algo.fit(trainset)
print('train finished.')
print()

# show factors matrix
pd.set_option('display.max_rows',10)
pd.set_option('display.max_columns',6)
print('Users factorization matrix is:')      # users factorization matrix
print(pd.DataFrame(algo.pu))                    # use pandas for printing purpose only
print()
print('Items factorization matrix is:')      # item factorization matrix
print(pd.DataFrame(algo.qi))
print()

# predict a single score
# algo.predict(192, 302, 4, verbose=True)

# show predicted score and actual score side by side
# randomly choose first 20 records with percentage 20/100000(# of records in total)
print('Randomly choose 1-10 records to compare side by side:')

limit = 10
for uid, iid, r, timestamp in data.raw_ratings:
    if random.random() > limit/100000:
        continue
    if limit == 0:
        break
    algo.predict(uid, iid, r, verbose=True)
    limit -= 1

print()

# show rmse
print('RMSE value of all rated scores is:')

predictions = []
for uid, iid, r, timestamp in data.raw_ratings:
    prediction = (r, algo.predict(uid, iid, r, verbose=False))
    predictions.append(prediction)
metrics.rmse(predictions, verbose=True)