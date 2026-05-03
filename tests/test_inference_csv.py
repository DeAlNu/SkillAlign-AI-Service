"""
Test Inference menggunakan CSV Test Cases
==========================================
Script ini melakukan batch testing pada model SkillAlign menggunakan
test cases dari file CSV untuk memastikan konsistensi dan akurasi prediksi.

Author: AI Engineer Team
Date: April 2026
Version: 1.0
"""

import sys
from pathlib import Path

# Tambahkan parent directory (SkillAlign-AI/) ke Python path
# Agar bisa import modul dari src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from typing import List, Dict
from dataclasses import dataclass
from src.inference.predict import SkillAlignPredictor


@dataclass
class TestResult:
    """Data class untuk menyimpan hasil setiap test case"""
    test_id: str
    description: str
    category: str
    expected_min: float
    expected_max: float
    actual_score: float
    passed: bool
    inference_time_ms: float
    confidence: str
    recommendation: str
    error: str = None


class CSVInferenceTester:
    """
    Class untuk menjalankan batch testing inference menggunakan CSV test cases
    """
    
    def __init__(self, 
                 model_path: str, 
                 preprocessor_path: str,
                 test_csv_path: str):
        """
        Initialize tester dengan path model, preprocessor, dan test CSV
        
        Args:
            model_path: Path ke file model .keras
            preprocessor_path: Path ke file preprocessor .pkl
            test_csv_path: Path ke file CSV test cases
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.test_csv_path = test_csv_path
        self.predictor = None
        self.results: List[TestResult] = []
        
    def load_predictor(self):
        """Load model dan preprocessor"""
        print("=" * 80)
        print("LOADING MODEL & PREPROCESSOR")
        print("=" * 80)
        
        start_time = time.time()
        self.predictor = SkillAlignPredictor(
            model_path=self.model_path,
            preprocessor_path=self.preprocessor_path
        )
        self.predictor.load()
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print(f"📍 Model path: {self.model_path}")
        print(f"📍 Preprocessor path: {self.preprocessor_path}")
        print()
        
    def load_test_cases(self) -> pd.DataFrame:
        """
        Load test cases dari CSV file
        
        Returns:
            DataFrame berisi test cases
        """
        print("=" * 80)
        print("LOADING TEST CASES")
        print("=" * 80)
        
        if not Path(self.test_csv_path).exists():
            raise FileNotFoundError(f"Test CSV file not found: {self.test_csv_path}")
        
        df = pd.read_csv(self.test_csv_path)
        
        # Validasi kolom yang diperlukan
        required_columns = ['test_id', 'cv_text', 'job_description', 
                          'expected_min_score', 'expected_max_score', 
                          'description', 'category']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        print(f"✅ Loaded {len(df)} test cases from CSV")
        print(f"📊 Categories: {df['category'].value_counts().to_dict()}")
        print()
        
        return df
    
    def run_single_test(self, row: pd.Series) -> TestResult:
        """
        Jalankan single test case
        
        Args:
            row: Single row dari DataFrame test cases
            
        Returns:
            TestResult object
        """
        test_id = row['test_id']
        cv_text = row['cv_text']
        job_description = row['job_description']
        expected_min = row['expected_min_score']
        expected_max = row['expected_max_score']
        description = row['description']
        category = row['category']
        
        try:
            # Run prediction
            start_time = time.time()
            result = self.predictor.predict(
                cv_text=cv_text,
                job_description=job_description
            )
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            actual_score = result.matching_score
            
            # Check if score is within expected range
            passed = expected_min <= actual_score <= expected_max
            
            return TestResult(
                test_id=test_id,
                description=description,
                category=category,
                expected_min=expected_min,
                expected_max=expected_max,
                actual_score=actual_score,
                passed=passed,
                inference_time_ms=inference_time,
                confidence=result.confidence,
                recommendation=result.recommendation
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                description=description,
                category=category,
                expected_min=expected_min,
                expected_max=expected_max,
                actual_score=-1.0,
                passed=False,
                inference_time_ms=-1.0,
                confidence="N/A",
                recommendation="N/A",
                error=str(e)
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """
        Jalankan semua test cases dari CSV
        
        Returns:
            List of TestResult objects
        """
        print("=" * 80)
        print("RUNNING BATCH INFERENCE TESTS")
        print("=" * 80)
        print()
        
        # Load test cases
        test_df = self.load_test_cases()
        
        # Run tests
        self.results = []
        total_tests = len(test_df)
        
        for idx, row in test_df.iterrows():
            print(f"[{idx + 1}/{total_tests}] Testing {row['test_id']}: {row['description']}")
            
            result = self.run_single_test(row)
            self.results.append(result)
            
            # Print immediate feedback
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {status} | Score: {result.actual_score:.4f} "
                  f"(Expected: {result.expected_min:.2f}-{result.expected_max:.2f}) "
                  f"| Time: {result.inference_time_ms:.1f}ms")
            
            if result.error:
                print(f"  ⚠️  Error: {result.error}")
            print()
        
        return self.results
    
    def generate_summary_report(self) -> Dict:
        """
        Generate summary report dari hasil testing
        
        Returns:
            Dictionary berisi summary statistics
        """
        if not self.results:
            raise ValueError("No results available. Run tests first.")
        
        results_df = pd.DataFrame([vars(r) for r in self.results])
        
        summary = {
            'total_tests': len(self.results),
            'passed_tests': int(results_df['passed'].sum()),
            'failed_tests': int((~results_df['passed']).sum()),
            'pass_rate': float(results_df['passed'].mean() * 100),
            'avg_inference_time_ms': float(results_df[results_df['inference_time_ms'] > 0]['inference_time_ms'].mean()),
            'max_inference_time_ms': float(results_df['inference_time_ms'].max()),
            'min_inference_time_ms': float(results_df[results_df['inference_time_ms'] > 0]['inference_time_ms'].min()),
            'category_breakdown': {},
            'failed_tests_details': []
        }
        
        # Category breakdown
        for category in results_df['category'].unique():
            cat_results = results_df[results_df['category'] == category]
            summary['category_breakdown'][category] = {
                'total': len(cat_results),
                'passed': int(cat_results['passed'].sum()),
                'pass_rate': float(cat_results['passed'].mean() * 100)
            }
        
        # Failed tests details
        failed = results_df[~results_df['passed']]
        for _, row in failed.iterrows():
            summary['failed_tests_details'].append({
                'test_id': row['test_id'],
                'description': row['description'],
                'expected_range': f"{row['expected_min']:.2f}-{row['expected_max']:.2f}",
                'actual_score': f"{row['actual_score']:.4f}",
                'error': row['error']
            })
        
        return summary
    
    def print_summary(self):
        """Print summary report ke console"""
        summary = self.generate_summary_report()
        
        print("=" * 80)
        print("TEST SUMMARY REPORT")
        print("=" * 80)
        print()
        
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']} ({summary['pass_rate']:.1f}%)")
        print(f"❌ Failed: {summary['failed_tests']}")
        print()
        
        print(f"⏱️  Average Inference Time: {summary['avg_inference_time_ms']:.1f}ms")
        print(f"⏱️  Max Inference Time: {summary['max_inference_time_ms']:.1f}ms")
        print(f"⏱️  Min Inference Time: {summary['min_inference_time_ms']:.1f}ms")
        print()
        
        print("📈 Performance by Category:")
        print("-" * 80)
        for category, stats in summary['category_breakdown'].items():
            status = "✅" if stats['pass_rate'] >= 80 else "⚠️" if stats['pass_rate'] >= 60 else "❌"
            print(f"{status} {category:20s} | Total: {stats['total']:2d} | "
                  f"Passed: {stats['passed']:2d} | Pass Rate: {stats['pass_rate']:5.1f}%")
        print()
        
        if summary['failed_tests_details']:
            print("❌ Failed Tests Details:")
            print("-" * 80)
            for fail in summary['failed_tests_details']:
                print(f"  • {fail['test_id']}: {fail['description']}")
                print(f"    Expected: {fail['expected_range']}, Actual: {fail['actual_score']}")
                if fail['error']:
                    print(f"    Error: {fail['error']}")
                print()
        
        print("=" * 80)
        
        # Overall status
        if summary['pass_rate'] >= 90:
            print("🎉 EXCELLENT! All critical tests passed!")
        elif summary['pass_rate'] >= 70:
            print("✅ GOOD! Most tests passed. Review failed cases.")
        else:
            print("⚠️  WARNING! Many tests failed. Model needs tuning.")
        print("=" * 80)
    
    def save_results_to_csv(self, output_path: str = 'tests/results/inference_test_results.csv'):
        """
        Save detailed results to CSV file
        
        Args:
            output_path: Path untuk menyimpan hasil testing
        """
        if not self.results:
            raise ValueError("No results available. Run tests first.")
        
        # Create directory if not exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([vars(r) for r in self.results])
        
        # Add timestamp
        results_df['test_timestamp'] = pd.Timestamp.now()
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Detailed results saved to: {output_path}")


def main():
    """Main function untuk menjalankan testing"""
    
    # Configuration - Gunakan path relatif dari root project
    MODEL_PATH = '../models/skillalign_matcher_v3.keras'
    PREPROCESSOR_PATH = '../preprocessors/nlp_preprocessor_v3.pkl'
    TEST_CSV_PATH = 'test_data/inference_test_cases.csv'
    OUTPUT_CSV_PATH = 'results/inference_test_results.csv'
    
    print("\n" + "=" * 80)
    print("SKILLALIGN INFERENCE TESTING SUITE")
    print("=" * 80)
    print()
    
    # Initialize tester
    tester = CSVInferenceTester(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        test_csv_path=TEST_CSV_PATH
    )
    
    try:
        # Load model
        tester.load_predictor()
        
        # Run all tests
        results = tester.run_all_tests()
        
        # Print summary
        tester.print_summary()
        
        # Save results
        tester.save_results_to_csv(OUTPUT_CSV_PATH)
        
        # Return exit code based on pass rate
        summary = tester.generate_summary_report()
        if summary['pass_rate'] >= 70:
            return 0  # Success
        else:
            return 1  # Failure
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())