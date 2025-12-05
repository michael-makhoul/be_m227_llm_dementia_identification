"""
Script to create a balanced test patient ID set.

This script creates a JSON file with patient IDs that are balanced
by dementia status (50% dementia, 50% non-dementia) and optionally
by demographics (gender, age, race/ethnicity) and year for testing purposes.

Balancing strategy:
- Dementia status: 50% with dementia, 50% without dementia
- Gender: Balanced within each dementia group (50/50 when possible)
- Year: Distributed across different years to avoid temporal bias
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Optional, List


def create_balanced_patient_set(
    csv_path: str,
    dem_col: str = 'expert_dem',
    n_patients: int = 10,
    random_seed: int = 42,
    output_file: str = 'test_patient_ids.json',
    balance_demographics: bool = True,
    demographic_vars: Optional[List[str]] = None
) -> dict:
    """
    Create a balanced patient ID set with 50% dementia and 50% non-dementia.
    Optionally balances by demographics (gender, age, race/ethnicity).
    
    Args:
        csv_path: Path to the CSV file with patient data
        dem_col: Column name for dementia status
        n_patients: Total number of patients to select (must be even)
        random_seed: Random seed for reproducibility
        output_file: Path to save the JSON file
        balance_demographics: Whether to balance by demographics
        demographic_vars: List of demographic variables to balance on
    
    Returns:
        Dictionary with patient IDs and metadata
    """
    np.random.seed(random_seed)
    
    if n_patients % 2 != 0:
        raise ValueError(f"n_patients must be even for 50/50 split. Got {n_patients}")
    
    n_per_group = n_patients // 2
    
    # Default demographic variables to balance
    if demographic_vars is None:
        demographic_vars = ['male', 'age_cont', 'race', 'hispanic']
    
    # Load the data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Get the most recent record for each patient
    print(f"Processing patient records...")
    
    # Aggregate columns - get last non-null value for each patient
    # Make this robust to CSVs that may not have a "row" or "year" column
    agg_dict = {
        dem_col: 'last',
    }
    if 'row' in df.columns:
        agg_dict['row'] = 'max'
    if 'year' in df.columns:
        agg_dict['year'] = 'last'  # Include year for balancing when available
    
    # Add demographic variables to aggregation
    for var in demographic_vars:
        if var in df.columns:
            agg_dict[var] = 'last'
    
    patient_data = df.groupby('hhidpn').agg(agg_dict).reset_index()
    
    # Clean the data - convert to numeric, handling NA and string values
    patient_data[dem_col] = pd.to_numeric(patient_data[dem_col], errors='coerce')
    
    # Remove patients with missing dementia status
    patient_data_clean = patient_data.dropna(subset=[dem_col]).copy()
    
    print(f"Patients with valid {dem_col} status: {len(patient_data_clean)}")
    print(f"Dementia status distribution:")
    print(patient_data_clean[dem_col].value_counts().sort_index())
    
    # Separate patients by dementia status
    no_dementia_df = patient_data_clean[patient_data_clean[dem_col] == 0].copy()
    with_dementia_df = patient_data_clean[patient_data_clean[dem_col] == 1].copy()
    
    print(f"\nAvailable patients without dementia: {len(no_dementia_df)}")
    print(f"Available patients with dementia: {len(with_dementia_df)}")
    
    if len(no_dementia_df) < n_per_group:
        raise ValueError(f"Not enough patients without dementia. Need {n_per_group}, have {len(no_dementia_df)}")
    if len(with_dementia_df) < n_per_group:
        raise ValueError(f"Not enough patients with dementia. Need {n_per_group}, have {len(with_dementia_df)}")
    
    # Function to sample with demographic and year balancing
    def sample_with_demographic_balance(df_group, n_samples):
        """Sample patients trying to balance demographics and year."""
        if not balance_demographics or n_samples <= 2:
            # Simple random sampling if not balancing or too few samples
            selected_ids = np.random.choice(df_group['hhidpn'].tolist(), size=n_samples, replace=False).tolist()
            return selected_ids, {}
        
        # Get available years
        available_years = sorted(df_group['year'].dropna().unique()) if 'year' in df_group.columns else []
        
        # Strategy: Try to balance by gender first, then distribute across years
        selected_ids = []
        demo_stats = {}
        
        if 'male' in df_group.columns and len(available_years) > 0:
            # Count gender distribution
            gender_counts = df_group['male'].value_counts()
            
            # Try to balance gender (50/50 if possible)
            n_male = min(n_samples // 2, gender_counts.get('Yes', 0) + gender_counts.get(1, 0))
            n_female = n_samples - n_male
            
            # Get male and female patients
            male_patients = df_group[df_group['male'].isin(['Yes', 1, '1', True])].copy()
            female_patients = df_group[~df_group['hhidpn'].isin(male_patients['hhidpn'])].copy()
            
            # If we don't have enough of one gender, use what we have
            if len(male_patients) < n_male:
                n_male = len(male_patients)
                n_female = n_samples - n_male
            if len(female_patients) < n_female:
                n_female = len(female_patients)
                n_male = n_samples - n_female
            
            # Distribute across years for each gender group
            def sample_with_year_balance(df_subset, n_needed):
                """Sample from subset trying to balance across years."""
                if len(df_subset) == 0:
                    return []
                
                selected = []
                year_counts = df_subset['year'].value_counts().to_dict()
                
                # Calculate target per year (roughly equal distribution)
                n_years = len(available_years)
                target_per_year = max(1, n_needed // n_years)
                remainder = n_needed % n_years
                
                # Sample from each year
                for i, year in enumerate(available_years):
                    year_patients = df_subset[df_subset['year'] == year]['hhidpn'].tolist()
                    if len(year_patients) == 0:
                        continue
                    
                    # Add one extra patient from first few years if needed
                    n_from_year = target_per_year + (1 if i < remainder else 0)
                    n_from_year = min(n_from_year, len(year_patients), n_needed - len(selected))
                    
                    if n_from_year > 0:
                        sampled = np.random.choice(year_patients, size=n_from_year, replace=False).tolist()
                        selected.extend(sampled)
                    
                    if len(selected) >= n_needed:
                        break
                
                # If we still need more, randomly sample from remaining
                if len(selected) < n_needed:
                    remaining = [pid for pid in df_subset['hhidpn'].tolist() if pid not in selected]
                    needed = n_needed - len(selected)
                    if len(remaining) >= needed:
                        additional = np.random.choice(remaining, size=needed, replace=False).tolist()
                        selected.extend(additional)
                
                return selected[:n_needed]
            
            # Sample males and females with year balancing
            selected_male = sample_with_year_balance(male_patients, n_male) if n_male > 0 else []
            selected_female = sample_with_year_balance(female_patients, n_female) if n_female > 0 else []
            
            selected_ids = selected_male + selected_female
            np.random.shuffle(selected_ids)
            
            # Get demographic stats
            selected_df = df_group[df_group['hhidpn'].isin(selected_ids)]
            demo_stats = {
                'gender_male': len(selected_df[selected_df['male'].isin(['Yes', 1, '1', True])]),
                'gender_female': len(selected_df) - len(selected_df[selected_df['male'].isin(['Yes', 1, '1', True])]),
            }
            
            if 'age_cont' in selected_df.columns:
                demo_stats['age_mean'] = float(selected_df['age_cont'].mean()) if selected_df['age_cont'].notna().any() else None
                demo_stats['age_std'] = float(selected_df['age_cont'].std()) if selected_df['age_cont'].notna().any() else None
            
            if 'year' in selected_df.columns:
                year_dist = selected_df['year'].value_counts().to_dict()
                demo_stats['year_distribution'] = {int(k): int(v) for k, v in year_dist.items()}
                demo_stats['years_represented'] = len(year_dist)
            
            return selected_ids, demo_stats
        elif 'male' in df_group.columns:
            # Balance by gender only (no year column)
            gender_counts = df_group['male'].value_counts()
            n_male = min(n_samples // 2, gender_counts.get('Yes', 0) + gender_counts.get(1, 0))
            n_female = n_samples - n_male
            
            male_patients = df_group[df_group['male'].isin(['Yes', 1, '1', True])]['hhidpn'].tolist()
            female_patients = df_group[~df_group['hhidpn'].isin(male_patients)]['hhidpn'].tolist()
            
            if len(male_patients) < n_male:
                n_male = len(male_patients)
                n_female = n_samples - n_male
            if len(female_patients) < n_female:
                n_female = len(female_patients)
                n_male = n_samples - n_female
            
            selected_male = np.random.choice(male_patients, size=n_male, replace=False).tolist() if n_male > 0 else []
            selected_female = np.random.choice(female_patients, size=n_female, replace=False).tolist() if n_female > 0 else []
            
            selected_ids = selected_male + selected_female
            np.random.shuffle(selected_ids)
            
            selected_df = df_group[df_group['hhidpn'].isin(selected_ids)]
            demo_stats = {
                'gender_male': len(selected_df[selected_df['male'].isin(['Yes', 1, '1', True])]),
                'gender_female': len(selected_df) - len(selected_df[selected_df['male'].isin(['Yes', 1, '1', True])]),
            }
            
            if 'age_cont' in selected_df.columns:
                demo_stats['age_mean'] = float(selected_df['age_cont'].mean()) if selected_df['age_cont'].notna().any() else None
                demo_stats['age_std'] = float(selected_df['age_cont'].std()) if selected_df['age_cont'].notna().any() else None
            
            return selected_ids, demo_stats
        elif len(available_years) > 0:
            # Balance by year only (no gender column)
            selected_ids = []
            n_years = len(available_years)
            target_per_year = max(1, n_samples // n_years)
            remainder = n_samples % n_years
            
            for i, year in enumerate(available_years):
                year_patients = df_group[df_group['year'] == year]['hhidpn'].tolist()
                if len(year_patients) == 0:
                    continue
                
                n_from_year = target_per_year + (1 if i < remainder else 0)
                n_from_year = min(n_from_year, len(year_patients), n_samples - len(selected_ids))
                
                if n_from_year > 0:
                    sampled = np.random.choice(year_patients, size=n_from_year, replace=False).tolist()
                    selected_ids.extend(sampled)
                
                if len(selected_ids) >= n_samples:
                    break
            
            # Fill remaining if needed
            if len(selected_ids) < n_samples:
                remaining = [pid for pid in df_group['hhidpn'].tolist() if pid not in selected_ids]
                needed = n_samples - len(selected_ids)
                if len(remaining) >= needed:
                    additional = np.random.choice(remaining, size=needed, replace=False).tolist()
                    selected_ids.extend(additional)
            
            selected_ids = selected_ids[:n_samples]
            np.random.shuffle(selected_ids)
            
            selected_df = df_group[df_group['hhidpn'].isin(selected_ids)]
            demo_stats = {}
            if 'age_cont' in selected_df.columns:
                demo_stats['age_mean'] = float(selected_df['age_cont'].mean()) if selected_df['age_cont'].notna().any() else None
                demo_stats['age_std'] = float(selected_df['age_cont'].std()) if selected_df['age_cont'].notna().any() else None
            
            year_dist = selected_df['year'].value_counts().to_dict()
            demo_stats['year_distribution'] = {int(k): int(v) for k, v in year_dist.items()}
            demo_stats['years_represented'] = len(year_dist)
            
            return selected_ids, demo_stats
        else:
            # No demographic columns, just random sample
            selected_ids = np.random.choice(df_group['hhidpn'].tolist(), size=n_samples, replace=False).tolist()
            return selected_ids, {}
    
    # Sample from each group
    print(f"\nSampling {n_per_group} patients from each group...")
    selected_no_dem, demo_stats_no_dem = sample_with_demographic_balance(no_dementia_df, n_per_group)
    selected_with_dem, demo_stats_with_dem = sample_with_demographic_balance(with_dementia_df, n_per_group)
    
    # Combine and shuffle
    all_selected = selected_no_dem + selected_with_dem
    np.random.shuffle(all_selected)
    
    # Get full demographic statistics for selected patients
    selected_df = patient_data_clean[patient_data_clean['hhidpn'].isin(all_selected)]
    
    # Create result dictionary
    result = {
        "n_patients": n_patients,
        "n_per_group": n_per_group,
        "dementia_column": dem_col,
        "random_seed": random_seed,
        "balance_demographics": balance_demographics,
        "patient_ids": [int(pid) for pid in all_selected],
        "metadata": {
            "no_dementia_ids": [int(pid) for pid in selected_no_dem],
            "with_dementia_ids": [int(pid) for pid in selected_with_dem],
            "total_available_no_dementia": len(no_dementia_df),
            "total_available_with_dementia": len(with_dementia_df),
            "demographics": {
                "no_dementia_group": demo_stats_no_dem,
                "with_dementia_group": demo_stats_with_dem,
                "overall": {
                    "gender_male": len(selected_df[selected_df['male'].isin(['Yes', 1, '1', True])]) if 'male' in selected_df.columns else None,
                    "gender_female": len(selected_df) - len(selected_df[selected_df['male'].isin(['Yes', 1, '1', True])]) if 'male' in selected_df.columns else None,
                    "age_mean": float(selected_df['age_cont'].mean()) if 'age_cont' in selected_df.columns and selected_df['age_cont'].notna().any() else None,
                    "age_std": float(selected_df['age_cont'].std()) if 'age_cont' in selected_df.columns and selected_df['age_cont'].notna().any() else None,
                }
            }
        }
    }
    
    # Add year distribution to overall stats if available
    if 'year' in selected_df.columns:
        year_dist = selected_df['year'].value_counts().to_dict()
        result['metadata']['demographics']['overall']['year_distribution'] = {int(k): int(v) for k, v in year_dist.items()}
        result['metadata']['demographics']['overall']['years_represented'] = len(year_dist)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Saved balanced patient ID set to {output_file}")
    print(f"  Total patients: {n_patients}")
    print(f"  Patients per group: {n_per_group}")
    print(f"  Balance demographics: {balance_demographics}")
    
    if balance_demographics and demo_stats_no_dem:
        print(f"\n  Demographics (No Dementia group):")
        print(f"    Male: {demo_stats_no_dem.get('gender_male', 'N/A')}, Female: {demo_stats_no_dem.get('gender_female', 'N/A')}")
        if 'age_mean' in demo_stats_no_dem and demo_stats_no_dem['age_mean']:
            print(f"    Age: {demo_stats_no_dem['age_mean']:.1f} ± {demo_stats_no_dem.get('age_std', 0):.1f}")
        if 'year_distribution' in demo_stats_no_dem:
            print(f"    Years represented: {demo_stats_no_dem.get('years_represented', 'N/A')}")
            print(f"    Year distribution: {demo_stats_no_dem.get('year_distribution', {})}")
        
        print(f"\n  Demographics (With Dementia group):")
        print(f"    Male: {demo_stats_with_dem.get('gender_male', 'N/A')}, Female: {demo_stats_with_dem.get('gender_female', 'N/A')}")
        if 'age_mean' in demo_stats_with_dem and demo_stats_with_dem['age_mean']:
            print(f"    Age: {demo_stats_with_dem['age_mean']:.1f} ± {demo_stats_with_dem.get('age_std', 0):.1f}")
        if 'year_distribution' in demo_stats_with_dem:
            print(f"    Years represented: {demo_stats_with_dem.get('years_represented', 'N/A')}")
            print(f"    Year distribution: {demo_stats_with_dem.get('year_distribution', {})}")
    
    if balance_demographics and 'year_distribution' in result['metadata']['demographics']['overall']:
        print(f"\n  Overall Year Distribution:")
        print(f"    Years represented: {result['metadata']['demographics']['overall'].get('years_represented', 'N/A')}")
        print(f"    Distribution: {result['metadata']['demographics']['overall'].get('year_distribution', {})}")
    
    print(f"\n  Selected Patient IDs: {result['patient_ids']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Create a balanced test patient ID set (50% dementia, 50% non-dementia)'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default='bm227_final_data.csv',
        help='Path to the CSV file with patient data (default: bm227_final_data.csv)'
    )
    parser.add_argument(
        '--dementia-column',
        type=str,
        default='expert_dem',
        help='Column name for dementia status (default: expert_dem)'
    )
    parser.add_argument(
        '--n-patients',
        type=int,
        default=10,
        help='Total number of patients to select (must be even, default: 10)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_patient_ids.json',
        help='Output JSON file path (default: test_patient_ids.json)'
    )
    parser.add_argument(
        '--no-demographic-balance',
        action='store_true',
        help='Disable demographic balancing (default: balance by gender and age)'
    )
    
    args = parser.parse_args()
    
    try:
        create_balanced_patient_set(
            csv_path=args.csv_path,
            dem_col=args.dementia_column,
            n_patients=args.n_patients,
            random_seed=args.random_seed,
            output_file=args.output,
            balance_demographics=not args.no_demographic_balance
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

