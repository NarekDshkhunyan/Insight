import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

qtype = pd.read_csv("Data/QuestionType.csv")         # 3846 x 3
tsummary = pd.read_csv("Data/TopicalSummary.csv")    # 5601 x 6

# Turn the varaibles into categorical and then use one-hot encoding
qtype['qual_cc'] = qtype['Qualitative Tag'].astype('category')
qtype['fund_cc'] = qtype['Fundamental Question'].astype('category')
qtype['qual_codes'] = qtype['qual_cc'].cat.codes
qtype['fund_codes'] = qtype['fund_cc'].cat.codes

# Plot the distribution of question tags
qtype = qtype.drop_duplicates() 
tags = qtype['qual_cc'].value_counts().keys()         # 58 unique tags, most frequent : General - 425/2842
values = qtype['qual_cc'].value_counts().tolist()

labels = [37, 17, 32, 54, 20, 0, 27, 33, 46, 3, 34, 2, 8, 14, 38, 30, 6, 41, 51, 4, 31, 10, 50, 26, 39, 49, 16, 1, 47, 36,
          55, 25, 57, 56, 9, 28, 15, 45, 42, 40, 21, 48, 52, 13, 24, 19, 5, 35, 22, 53, 18, 44, 7, 12, 43, 11, 23, 29]
labels = np.array(labels)

plt.bar(np.arange(len(values)), values, align='center', alpha=0.5, color='blue')
plt.xticks(np.arange(labels), labels)
plt.ylabel('Number of tags')
plt.title("Question tags")

# Make a copy of the csv, with only the tags and corresponding label
copy = qtype.drop(['Fundamental Question', 'qual_cc', 'fund_cc', 'fund_codes'], axis=1)
copy.to_csv('TagsLabels.csv', sep='\t')

shuffledData = copy.reindex(np.random.permutation(copy.index))
shuffledData.to_csv('Shuffled.csv', sep='\t')
