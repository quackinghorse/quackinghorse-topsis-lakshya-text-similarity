import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv', index_col='Model')

normalized_data = data.div(np.sqrt((data**2).sum(axis=0)), axis=1)

weights = {'Accuracy': 0.4, 'Speed': 0.3, 'Versatility': 0.3}

weighted_normalized_data = normalized_data * np.array(list(weights.values()))

positive_ideal = weighted_normalized_data.max()
negative_ideal = weighted_normalized_data.min()

positive_distance = np.sqrt(((weighted_normalized_data - positive_ideal)**2).sum(axis=1))
negative_distance = np.sqrt(((weighted_normalized_data - negative_ideal)**2).sum(axis=1))

closeness = negative_distance / (positive_distance + negative_distance)

data['Topsis Score'] = closeness

data['Rank'] = data['Topsis Score'].rank(ascending=False)

result = data[['Topsis Score', 'Rank']].sort_values(by='Rank')
result.to_csv('result.csv')

plt.figure(figsize=(10, 6))
result['Topsis Score'].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Topsis Score for Text Sentence Similarity Models')
plt.xlabel('Model')
plt.ylabel('Topsis Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('topsis_score_plot.png')

# Display the plot
plt.show()