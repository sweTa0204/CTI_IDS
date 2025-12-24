#!/usr/bin/env python3
"""
Verification Script: Analyze removed features and validate smart selection decisions
"""

import pandas as pd
import numpy as np
from scipy import stats

def main():
    print('=' * 60)
    print('🔍 SMART SELECTION DECISION VALIDATION')
    print('=' * 60)
    print('Analyzing if our feature removal decisions were correct...')
    print()

    # Load original and decorrelated datasets
    print('📂 Loading datasets for comparison...')
    df_original = pd.read_csv('../data/encoded_dataset.csv')
    df_decorrelated = pd.read_csv('../data/decorrelated_dataset.csv')
    
    # Features that were removed
    removed_features = ['tcprtt', 'sbytes', 'sloss', 'is_sm_ips_ports', 
                       'dpkts', 'dloss', 'is_ftp_login', 'dwin']
    
    # Features that were kept (their correlates)
    kept_correlates = {
        'tcprtt': 'synack',      # tcprtt removed, synack kept
        'sbytes': 'spkts',       # sbytes removed, spkts kept  
        'sloss': 'sbytes',       # sloss removed, sbytes kept (but sbytes also removed!)
        'is_sm_ips_ports': 'sinpkt',  # is_sm_ips_ports removed, sinpkt kept
        'dpkts': 'dbytes',       # dpkts removed, dbytes kept
        'dloss': 'dbytes',       # dloss removed, dbytes kept
        'is_ftp_login': 'ct_ftp_cmd',  # is_ftp_login removed, ct_ftp_cmd kept
        'dwin': 'swin'           # dwin removed, swin kept
    }
    
    print('📊 ANALYZING EACH REMOVAL DECISION:')
    print()
    
    # Separate DoS and Normal data
    dos_data = df_original[df_original['label'] == 1]
    normal_data = df_original[df_original['label'] == 0]
    
    for removed_feat in removed_features:
        kept_feat = kept_correlates[removed_feat]
        
        print(f"🔍 DECISION: Removed '{removed_feat}' vs Kept '{kept_feat}'")
        
        # Check correlation
        corr = df_original[removed_feat].corr(df_original[kept_feat])
        print(f"   Correlation: {corr:.3f}")
        
        # Check variance
        removed_var = df_original[removed_feat].var()
        kept_var = df_original[kept_feat].var()
        print(f"   Variance - Removed: {removed_var:.6f}, Kept: {kept_var:.6f}")
        
        # Check DoS vs Normal discrimination (ANOVA F-test)
        removed_f_stat, removed_p_val = stats.f_oneway(
            dos_data[removed_feat], normal_data[removed_feat]
        )
        kept_f_stat, kept_p_val = stats.f_oneway(
            dos_data[kept_feat], normal_data[kept_feat]
        )
        
        print(f"   DoS vs Normal F-statistic - Removed: {removed_f_stat:.2f}, Kept: {kept_f_stat:.2f}")
        print(f"   DoS vs Normal p-value - Removed: {removed_p_val:.6f}, Kept: {kept_p_val:.6f}")
        
        # Check means for DoS vs Normal
        removed_dos_mean = dos_data[removed_feat].mean()
        removed_normal_mean = normal_data[removed_feat].mean()
        kept_dos_mean = dos_data[kept_feat].mean()
        kept_normal_mean = normal_data[kept_feat].mean()
        
        print(f"   DoS Mean - Removed: {removed_dos_mean:.3f}, Kept: {kept_dos_mean:.3f}")
        print(f"   Normal Mean - Removed: {removed_normal_mean:.3f}, Kept: {kept_normal_mean:.3f}")
        
        # Decision validation
        better_discrimination = kept_f_stat > removed_f_stat
        better_variance = kept_var > removed_var
        more_significant = kept_p_val < removed_p_val
        
        print(f"   ✅ Better discrimination: {better_discrimination}")
        print(f"   ✅ Better variance: {better_variance}")
        print(f"   ✅ More significant: {more_significant}")
        
        # Overall decision quality
        good_decision = sum([better_discrimination, better_variance, more_significant]) >= 2
        print(f"   🎯 Decision Quality: {'GOOD ✅' if good_decision else 'QUESTIONABLE ⚠️'}")
        print()
    
    # Special analysis for problematic cases
    print('🚨 SPECIAL ANALYSIS: Potential Issues')
    print()
    
    # Check sbytes vs sloss issue
    print("🔍 ISSUE: sbytes-sloss chain removal")
    print("   Original correlation: sbytes ↔ sloss (0.986)")
    print("   Decision: Remove sloss, keep sbytes")
    print("   BUT: sbytes was later removed due to spkts ↔ sbytes (0.954)")
    print("   Result: Both sbytes AND sloss were removed!")
    
    # Check if we lost important information
    if 'sloss' not in df_decorrelated.columns and 'sbytes' not in df_decorrelated.columns:
        print("   ⚠️  WARNING: Lost both source bytes and source loss information!")
        print("   ⚠️  Only have spkts (source packets) remaining")
        
        # Check if spkts adequately represents the information
        spkts_sbytes_corr = df_original['spkts'].corr(df_original['sbytes'])
        spkts_sloss_corr = df_original['spkts'].corr(df_original['sloss'])
        print(f"   📊 spkts ↔ sbytes correlation: {spkts_sbytes_corr:.3f}")
        print(f"   📊 spkts ↔ sloss correlation: {spkts_sloss_corr:.3f}")
        
        # Check discrimination power
        spkts_f_stat, spkts_p_val = stats.f_oneway(
            dos_data['spkts'], normal_data['spkts']
        )
        sbytes_f_stat, sbytes_p_val = stats.f_oneway(
            dos_data['sbytes'], normal_data['sbytes']
        )
        
        print(f"   📈 Discrimination power:")
        print(f"      spkts F-stat: {spkts_f_stat:.2f}, p-val: {spkts_p_val:.6f}")
        print(f"      sbytes F-stat: {sbytes_f_stat:.2f}, p-val: {sbytes_p_val:.6f}")
        
        if sbytes_f_stat > spkts_f_stat * 1.1:  # 10% better
            print("   ⚠️  WARNING: sbytes has significantly better discrimination!")
            print("   💡 RECOMMENDATION: Consider keeping sbytes instead of spkts")
    
    print()
    print('=' * 60)
    print('📋 VALIDATION SUMMARY')
    print('=' * 60)
    
    # Overall assessment
    print('🎯 OVERALL ASSESSMENT:')
    print()
    
    remaining_features = df_decorrelated.drop('label', axis=1).columns.tolist()
    print(f'Features before correlation removal: {len(df_original.columns) - 1}')
    print(f'Features after correlation removal: {len(remaining_features)}')
    print(f'Features removed: {len(removed_features)}')
    print()
    
    print('✅ GOOD DECISIONS:')
    print('1. dloss → dbytes: dloss is redundant with destination bytes')
    print('2. dpkts → dbytes: destination packets redundant with bytes')
    print('3. is_ftp_login → ct_ftp_cmd: FTP login flag redundant with command count')
    print('4. dwin → swin: destination window redundant with source window')
    print()
    
    print('⚠️  QUESTIONABLE DECISIONS:')
    print('1. tcprtt vs synack: Both are timing features, need to verify which is better')
    print('2. sbytes removal: Lost both sbytes AND sloss, might lose important info')
    print('3. is_sm_ips_ports: Might be removing useful security feature')
    print()
    
    print('💡 RECOMMENDATIONS:')
    print('1. ✅ Keep current removals for clearly redundant pairs (dloss, dpkts, etc.)')
    print('2. 🔍 Review timing features (tcprtt vs synack)')
    print('3. 🔍 Review source traffic representation (spkts vs sbytes)')
    print('4. ✅ Proceed with variance analysis to remove truly uninformative features')

if __name__ == "__main__":
    main()
