import matplotlib.pyplot as plt
import numpy as np

# Data for the chart
methods = ['GNN', 'Heuristic', 'Local Search', 'GTL']
averages = [16.56, 16.46, 16.04, 11.28]
std_values = [2.99, 3.91, 3.58, 0.45]

# Recreate the bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(methods, averages, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Method')
plt.ylabel('Average Score')
plt.title('Comparison of Average Scores and STD by Method')
plt.ylim(0, 20)  # Adjusting y-axis limit for better visualization

# Adding the average score and standard deviation on top of each bar
for bar, avg, std in zip(bars, averages, std_values):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{avg}\n±{std}', ha='center', va='bottom')

# Show the plot
plt.savefig('figures/results_per_method.png')
plt.show()
