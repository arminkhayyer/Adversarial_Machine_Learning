import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
df = pd.read_csv('casis25_ncu.txt', header=None)
print(df.shape)
features = ['casis25_char-gram_gram=3-limit=1000.txt', 'casis25_bow.txt', 'casis25_sty.txt']

for feature in features:
    df_feature = pd.read_csv(feature, header=None)
    df = pd.merge(df, df_feature, on=0, how="left")
    print(df_feature.shape)
    print('adding {}'.format(feature))

print(df)
# print(df.shape)
labels = df[0].map(lambda x: str(x)[0:4])
labels = pd.get_dummies(labels, prefix=None,drop_first=False)
df = df.drop(df.columns[[0]], axis=1)
X = df.to_numpy()
Y = labels.to_numpy()
Y= Y.astype(float)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# X[: , 0:95]  Unigram features
# X[:, 95: 1095]  3_gram, limit=1000 features
# X[:, 1095: 7213]  BagOfWords features
# X[:, 7213: ] Stylomerty features
# Y represents as one-hot encoding

np.save('X_train.npy',X_train)
np.save('Y_train.npy',Y_train)
np.save('X_test.npy',X_test)
np.save('Y_test.npy',Y_test)