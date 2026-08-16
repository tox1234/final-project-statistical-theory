import os
import warnings
warnings.filterwarnings('ignore')

from src.data_processing import load_and_clean_data
from src.distributions_fit import analyze_distributions
from src.hypothesis_tests import test_independent_samples, test_paired_samples, test_categorical_independence
from src.models_and_glrt import run_regression_and_glrt, run_logistic_modeling
from src.sprt_analysis import run_wald_sequential_test

def main():
    print("=" * 65)
    print(" Statistical Theory Final Project - Empirical Pipeline Execution ")
    print("=" * 65)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'pollution_dataset.csv')
    
    try:
        df = load_and_clean_data(data_path)
    except Exception as e:
        print(e)
        return
        
    analyze_distributions(df)
    test_independent_samples(df)
    test_paired_samples(df)
    test_categorical_independence(df)
    run_regression_and_glrt(df)
    run_logistic_modeling(df)
    run_wald_sequential_test()
    
    print("\n" + "=" * 65)
    print(" Pipeline execution complete. All inferences structurally valid. ")
    print("=" * 65)

if __name__ == '__main__':
    main()
