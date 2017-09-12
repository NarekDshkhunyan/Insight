import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

qtype = pd.read_csv("Question Type.csv")         # 3846 x 3
tsummary = pd.read_csv("Topical Summary.csv")    # 5601 x 6

# Turn the varaibles into categorical and then use one-hot encoding
qtype['qual_cc'] = qtype['Qualitative Tag'].astype('category')
qtype['fund_cc'] = qtype['Fundamental Question'].astype('category')
qtype['qual_codes'] = qtype['qual_cc'].cat.codes
qtype['fund_codes'] = qtype['fund_cc'].cat.codes


# Plot the distribution of question tags
tags = qtype['Categorial Type'].value_counts().keys()         # 58 unique tags
values = qtype['Categorial Type'].value_counts().tolist()

plt.bar(np.arange(len(values)), values, align='center', alpha=0.5, color='blue')
plt.xticks(np.arange(len(tags)), tags)
plt.ylabel('Number of tags')
plt.title("Question tags")

# Plot the distribution of fundamental questions
tags = qtype['Fundamental Question'].value_counts().keys()     # 259 unique fundamental questions
values = qtype['Fundamental Question'].value_counts().tolist()

plt.bar(np.arange(len(values)), values, align='center', alpha=0.5, color='blue')
plt.xticks(np.arange(len(tags)), tags)
plt.ylabel('Number of questions')
plt.title("Fundamental questions")
