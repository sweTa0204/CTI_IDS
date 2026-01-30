# Step 1: DoS Detection Dataset Creation

## Overview
This step creates a **balanced binary classification dataset** for DoS attack detection by extracting both DoS attacks and Normal traffic from the UNSW-NB15 dataset. This is the foundation for building an effective DoS detection system.

## Understanding the UNSW-NB15 Dataset

### What is UNSW-NB15?
The UNSW-NB15 dataset is a comprehensive **network security dataset** created by the University of New South Wales, Australia. Think of it as a massive collection of internet traffic records - like having a security camera that records every single data packet flowing through a network.

### Complete Dataset Structure
The original UNSW-NB15 training set contains **10 different categories**:

| Category | Records | Percentage | Description |
|----------|---------|------------|-------------|
| **Normal** | 37,000 | 44.94% | **Legitimate network traffic** (web browsing, email, file downloads) |
| **Generic** | 18,871 | 22.92% | Generic attack patterns |
| **Exploits** | 11,132 | 13.52% | Software exploitation attacks |
| **Fuzzers** | 6,062 | 7.36% | Fuzzing and input validation attacks |
| **DoS** | 4,089 | 4.97% | **Denial of Service attacks (Our Focus!)** |
| **Reconnaissance** | 3,496 | 4.25% | Information gathering attacks |
| **Analysis** | 677 | 0.82% | Network analysis attacks |
| **Backdoor** | 583 | 0.71% | Backdoor installation attempts |
| **Shellcode** | 378 | 0.46% | Shellcode injection attacks |
| **Worms** | 44 | 0.05% | Worm propagation attempts |

**Total**: 82,332 records with 45 features

### Key Columns Explained
- **`attack_cat`**: Text labels ("Normal", "DoS", "Exploits", etc.)
- **`label`**: Binary indicators (0 = Normal, 1 = Any Attack)
- **`id`**: Row identifiers (not useful for machine learning)
- **42 Feature Columns**: Network traffic characteristics (duration, protocols, packet counts, etc.)

## Our Project Focus: Binary DoS Detection

### Why Binary Classification?
Instead of detecting all 10 attack types, we focus specifically on **DoS vs Normal** classification because:
- **Real-world relevance**: DoS attacks are critical security threats
- **Clear binary problem**: Either traffic is legitimate or it's a DoS attack
- **Specialized detection**: Different attack types require different detection approaches
- **Better performance**: Binary models typically perform better than multi-class models

### The Class Imbalance Problem

**If we used all available data:**
```
Normal Traffic: 37,000 records (90.1%)
DoS Attacks:     4,089 records (9.9%)
Total:          41,089 records
```

**Why this is problematic:**
- **Model bias**: The model would learn Normal patterns really well but barely learn DoS patterns
- **Poor DoS detection**: Model might predict "everything is Normal" and still be 90% accurate
- **Security risk**: Missing DoS attacks is unacceptable for security systems

**Real-world analogy**: Learning to recognize cats vs dogs with 900 cat pictures and only 100 dog pictures - you'd become great at recognizing cats but terrible at recognizing dogs!

### Our Solution: Balanced Dataset

**Our balanced approach:**
```
Normal Traffic: 4,089 records (50%)
DoS Attacks:    4,089 records (50%)
Total:          8,178 records
```

**Benefits of 50/50 balance:**
- **Equal learning opportunity**: Model sees equal examples of both classes
- **Unbiased training**: No preference toward majority class
- **Better security performance**: Model becomes equally good at detecting Normal and DoS traffic
- **Meaningful metrics**: Accuracy becomes a reliable performance indicator

## Normal Traffic Sampling Strategy

### How We Selected 4,089 Normal Records

**Question**: Why use 4,089 Normal records instead of all 37,000?

**Answer**: We used **random sampling** to create a balanced dataset:

```python
# From 37,000 Normal records, randomly select 4,089
normal_traffic = normal_traffic.sample(n=4089, random_state=42)
```

### Sampling Method Analysis

**What we did**: **Pure Random Sampling**
- **Method**: Completely random selection from 37,000 Normal records
- **No conditions**: No specific criteria or stratification applied
- **Reproducible**: `random_state=42` ensures same results every time

**Assessment of our random sample**:

**Protocol Diversity** (What we actually got):
- **TCP**: 3,107 records (76%) - Web traffic, email, file transfers
- **UDP**: 861 records (21%) - DNS, streaming, real-time communications
- **ARP**: 118 records (3%) - Network address resolution
- **OSPF**: 2 records (<1%) - Routing protocol
- **IGMP**: 1 record (<1%) - Multicast protocol

**Service Diversity** (What we actually got):
- **Generic (-)**: 3,031 records (74%) - Various background traffic
- **HTTP**: 435 records (11%) - Web browsing
- **DNS**: 335 records (8%) - Domain name resolution
- **FTP**: 208 records (5%) - File transfer
- **SMTP**: 59 records (1%) - Email sending
- **SSH**: 19 records (<1%) - Secure remote access

**Conclusion**: **Our random sampling worked well!** We got good diversity across protocols and services, representing realistic network traffic patterns.

### Alternative Sampling Methods (Not Used)

**1. Stratified Sampling**: Maintain original protocol proportions
**2. Balanced Service Sampling**: Equal samples from each service type
**3. Cluster-Based Sampling**: Group similar traffic patterns and sample from each
**4. Time-Based Sampling**: Sample from different time periods
**5. Feature-Based Sampling**: Stratify based on network characteristics

**Why we didn't use them**: Our random sampling achieved good diversity, and more complex methods would require significantly more effort for marginal improvement.

## Process Details

### 1. Dataset Loading and Validation
```python
# Load complete UNSW-NB15 training dataset
df = pd.read_csv('UNSW_NB15_training-set.csv')
# Result: 82,332 records × 45 features
```

### 2. Attack Category Analysis
```python
# Analyze distribution of all attack categories
attack_dist = df['attack_cat'].value_counts()
# Identified: 37,000 Normal + 4,089 DoS + 41,243 other attacks
```

### 3. Balanced Extraction
```python
# Extract DoS attacks (all available)
dos_attacks = df[df['attack_cat'] == 'DoS'].copy()  # 4,089 records

# Extract Normal traffic (randomly sampled)
normal_traffic = df[df['attack_cat'] == 'Normal'].copy()  # 37,000 available
normal_sample = normal_traffic.sample(n=4089, random_state=42)  # 4,089 selected

# Combine and shuffle
balanced_dataset = pd.concat([dos_attacks, normal_sample], ignore_index=True)
balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
```

### 4. Data Quality Assessment
- **Missing Values**: 0 (perfect data quality)
- **Duplicate Records**: 0 (no duplicates found)
- **Data Type Consistency**: Maintained across all features
- **Feature Preservation**: All 45 original features retained

## Output

### Primary Output
- **File**: `dos_detection_dataset.csv`
- **Records**: 8,178 (4,089 DoS + 4,089 Normal)
- **Features**: 45 (including id, attack_cat, label)
- **Balance**: Perfect 50/50 split
- **Location**: `../data/`

### Secondary Outputs
- **Feature Info**: `feature_info.csv` - detailed metadata for all 42 input features
- **Analysis Report**: `step1_dos_detection_extraction_report.txt` - comprehensive summary
- **Quality Metrics**: Missing values, duplicates, data types analysis

## Key Statistics

### Dataset Composition
- **Total Records**: 8,178
- **DoS Attacks**: 4,089 (50.00%)
- **Normal Traffic**: 4,089 (50.00%)
- **Features for ML**: 42 (excluding id, attack_cat, label)

### Data Quality Metrics
- **Missing Values**: 0 (100% complete data)
- **Duplicate Records**: 0 (100% unique records)
- **Data Types**: 30 int64, 11 float64, 4 object (categorical)
- **File Size**: 1.48 MB

### Feature Categories
- **Numeric Features**: 39 (ready for ML algorithms)
- **Categorical Features**: 3 (proto, service, state - need encoding)
- **Target Variables**: 2 (attack_cat, label - choose one)

## Why This Approach is Superior

### Compared to Imbalanced Dataset
| Aspect | Imbalanced (37K Normal, 4K DoS) | Balanced (4K Normal, 4K DoS) |
|--------|----------------------------------|-------------------------------|
| **Model Bias** | Heavy bias toward Normal | No bias |
| **DoS Detection** | Poor (often missed) | Excellent |
| **False Positives** | Low but misleading | Realistic rate |
| **Security Value** | Low (misses attacks) | High (reliable detection) |
| **Training Efficiency** | Inefficient | Efficient |

### Real-World Benefits
1. **Better Security**: Model reliably detects DoS attacks
2. **Balanced Performance**: Equal accuracy on both Normal and DoS traffic
3. **Practical Deployment**: Ready for real network security systems
4. **Interpretable Results**: Performance metrics are meaningful and trustworthy

## Next Steps

The balanced DoS detection dataset enables **Step 2: Feature Engineering** with six sub-steps:

### Step 2.1: Data Cleanup
- Remove unnecessary columns (id)
- Choose target variable (attack_cat vs label)
- Prepare clean feature matrix

### Step 2.2: Categorical Encoding
- Convert text features (proto, service, state) to numeric
- Enable machine learning algorithm compatibility

### Step 2.3: Correlation Analysis
- Identify and remove highly correlated features
- Reduce redundancy and complexity

### Step 2.4: Variance Analysis
- Remove low-variance features
- Eliminate uninformative variables

### Step 2.5: Statistical Testing
- Test feature discrimination power (DoS vs Normal)
- Use ANOVA F-tests and mutual information scoring

### Step 2.6: Final Feature Selection
- Combine all analysis results
- Select 12-15 most important features for final model

## Success Criteria

### Quality Assurance Achieved
- ✅ **Balanced Classes**: Perfect 50/50 DoS/Normal split
- ✅ **Data Integrity**: No missing values or duplicates
- ✅ **Feature Preservation**: All 42 input features maintained
- ✅ **Sampling Quality**: Good diversity across protocols and services
- ✅ **Reproducibility**: Random seed ensures consistent results

### Ready for Machine Learning
- ✅ **Binary Classification**: Clear DoS vs Normal problem
- ✅ **Sufficient Size**: 8,178 records provide statistical power
- ✅ **Representative Sample**: Covers diverse network traffic patterns
- ✅ **Clean Foundation**: High-quality base for feature engineering

This balanced dataset provides the optimal foundation for building an effective DoS detection system that can reliably distinguish between legitimate network traffic and DoS attacks in real-world scenarios.
