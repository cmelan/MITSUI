import pandas as pd
import numpy as np

# Load the complete test data
test_data = pd.read_csv('data/test.csv')
print(f"📊 Test data shape: {test_data.shape}")
print(f"📅 Test date range: {test_data['date_id'].min()} to {test_data['date_id'].max()}")

# Create submission with all test dates
date_ids = test_data['date_id'].tolist()
submission_data = {'date_id': date_ids}

# Add all 424 targets with reasonable values
for i in range(424):
    target_col = f'target_{i}'
    # Use small random values around 0 (typical for financial returns)
    submission_data[target_col] = np.random.normal(0, 0.01, len(date_ids))

# Create the submission dataframe
submission_df = pd.DataFrame(submission_data)

# Ensure all values are finite
for col in submission_df.columns:
    if col.startswith('target_'):
        submission_df[col] = submission_df[col].fillna(0)
        submission_df[col] = submission_df[col].replace([np.inf, -np.inf], 0)

# Save the submission file
submission_df.to_csv('outputs/complete_submission.csv', index=False, encoding='utf-8')

print(f"✅ Created complete submission file")
print(f"📊 Shape: {submission_df.shape}")
print(f"🎯 Target columns: {len([col for col in submission_df.columns if 'target' in col])}")
print(f"📅 Date range: {submission_df['date_id'].min()} to {submission_df['date_id'].max()}")
print(f"📈 Target value range: {submission_df[[col for col in submission_df.columns if 'target' in col]].min().min():.6f} to {submission_df[[col for col in submission_df.columns if 'target' in col]].max().max():.6f}")
print(f"💾 Saved to: outputs/complete_submission.csv")

# Save a minimal submission file (header + first row)
minimal_df = submission_df.head(1)
minimal_df.to_csv('outputs/minimal_submission.csv', index=False, encoding='utf-8')
print(f"✅ Created minimal submission file")
print(f"📊 Shape: {minimal_df.shape}")
print(f"🎯 Target columns: {len([col for col in minimal_df.columns if 'target' in col])}")
print(f"📅 Date range: {minimal_df['date_id'].min()} to {minimal_df['date_id'].max()}")
print(f"📈 Target value range: {minimal_df[[col for col in minimal_df.columns if 'target' in col]].min().min():.6f} to {minimal_df[[col for col in minimal_df.columns if 'target' in col]].max().max():.6f}")
print(f"�� Saved to: outputs/minimal_submission.csv") 