import pandas as pd
 
df = pd.read_csv('private_clustered_data.csv')
submission = df[['id', 'Cluster']].rename(columns={'Cluster': 'label'})
submission.to_csv('private_submission.csv', index=False)
print('private_submission.csv created.') 