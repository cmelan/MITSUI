import pandas as pd
import numpy as np

# Create a very simple submission with just a few rows
test_data = pd.read_csv('data/test.csv')
date_ids = test_data['date_id'].head(5).tolist()  # Just first 5 rows

# Create minimal submission
submission_data = {'date_id': date_ids}

# Add all 424 targets with simple values
for i in range(424):
    target_col = f'target_{i}'
    submission_data[target_col] = [0.0] * len(date_ids)  # All zeros

# Create the submission dataframe
submission_df = pd.DataFrame(submission_data)

# Save the submission file
submission_df.to_csv('outputs/test_submission.csv', index=False)

print(f"✅ Created test submission file")
print(f"📊 Shape: {submission_df.shape}")
print(f"🎯 Target columns: {len([col for col in submission_df.columns if 'target' in col])}")
print(f"📅 Date range: {submission_df['date_id'].min()} to {submission_df['date_id'].max()}")
print(f"💾 Saved to: outputs/test_submission.csv") 