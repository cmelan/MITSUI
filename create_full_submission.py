import pandas as pd
import numpy as np

# Load the test data to get the date_ids
test_data = pd.read_csv('data/test.csv')
date_ids = test_data['date_id'].tolist()

# Create a submission dataframe with all 424 targets
submission_data = {'date_id': date_ids}

# Add all 424 target columns (target_0 to target_423)
for i in range(424):
    target_col = f'target_{i}'
    # Generate random predictions for now (you can replace with actual model predictions)
    submission_data[target_col] = np.random.normal(0, 1, len(date_ids))

# Create the submission dataframe
submission_df = pd.DataFrame(submission_data)

# Save the submission file
submission_df.to_csv('outputs/submission.csv', index=False)

print(f"✅ Created submission file with {len(submission_df.columns)} columns")
print(f"📊 Shape: {submission_df.shape}")
print(f"🎯 Target columns: {len([col for col in submission_df.columns if 'target' in col])}")
print(f"📅 Date range: {submission_df['date_id'].min()} to {submission_df['date_id'].max()}")
print(f"💾 Saved to: outputs/submission.csv") 