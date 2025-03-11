import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON data
with open('evaluation_results.json', 'r') as file:
    data = json.load(file)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Set up the matplotlib figure
fig = plt.figure(figsize=(14, 10))

# Create a list to store legend handles and labels
handles, labels = None, None

# Plotting MAE
ax1 = plt.subplot(2, 3, 1)
sns.barplot(x='preprocessing', y='MAE', hue='model', data=df, ax=ax1)
plt.title('MAE by Preprocessing and Model')
plt.xticks(rotation=45)
# Remove the legend from this subplot
ax1.get_legend().remove()
# Save handles and labels for later use
handles, labels = ax1.get_legend_handles_labels()

# Plotting MSE
ax2 = plt.subplot(2, 3, 2)
sns.barplot(x='preprocessing', y='MSE', hue='model', data=df, ax=ax2)
plt.title('MSE by Preprocessing and Model')
plt.xticks(rotation=45)
# Remove the legend
ax2.get_legend().remove()

# Plotting RMSE
ax3 = plt.subplot(2, 3, 3)
sns.barplot(x='preprocessing', y='RMSE', hue='model', data=df, ax=ax3)
plt.title('RMSE by Preprocessing and Model')
plt.xticks(rotation=45)
# Remove the legend
ax3.get_legend().remove()

# Plotting R2 Score
ax4 = plt.subplot(2, 3, 4)
sns.barplot(x='preprocessing', y='R2_score', hue='model', data=df, ax=ax4)
plt.title('R2 Score by Preprocessing and Model')
plt.xticks(rotation=45)
# Remove the legend
ax4.get_legend().remove()

# Plotting Relative Error
ax5 = plt.subplot(2, 3, 5)
sns.barplot(x='preprocessing', y='Relative Error', hue='model', data=df, ax=ax5)
plt.title('Relative Error by Preprocessing and Model')
plt.xticks(rotation=45)
# Remove the legend
ax5.get_legend().remove()

# Create a single legend in the bottom right empty space (position 6)
ax6 = plt.subplot(2, 3, 6)
# Hide the axis for this subplot
ax6.axis('off')
# Add the legend to this empty subplot
ax6.legend(handles, labels, loc='center')

# Adjust layout
plt.tight_layout()
plt.show()