import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDA_Analysis:
    def __init__(self, train_file, test_file):
        # Load train and test datasets
        logging.info("Loading datasets...")
        self.df_train = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)

        # Strip any leading/trailing whitespace from column names
        self.df_train.columns = self.df_train.columns.str.strip()
        self.df_test.columns = self.df_test.columns.str.strip()
        
        logging.info("Train columns: %s", self.df_train.columns.tolist())
        logging.info("Test columns: %s", self.df_test.columns.tolist())

        # Clean the Date column and convert to datetime
        self.df_train['Date'] = pd.to_datetime(self.df_train['Date'], errors='coerce')
        self.df_test['Date'] = pd.to_datetime(self.df_test['Date'], errors='coerce')

        # Drop rows where Date could not be converted
        self.df_train.dropna(subset=['Date'], inplace=True)
        self.df_test.dropna(subset=['Date'], inplace=True)

        # Check if 'Store' is present in both datasets
        if 'Store' in self.df_train.columns and 'Store' in self.df_test.columns:
            logging.info("Merging datasets on 'Store'...")
            self.df_merged = pd.merge(self.df_train, self.df_test[['Store', 'Promo']], on='Store', how='left')
        else:
            logging.error("Both datasets must contain the 'Store' column for merging.")
            raise ValueError("Both datasets must contain the 'Store' column for merging.")

    def promo_distribution_comparison(self, save_path='../Images/promo_distribution_comparison.png'):
        logging.info("Comparing promo distribution between training and test sets...")
        train_promo_dist = self.df_train['Promo'].value_counts(normalize=True) * 100
        test_promo_dist = self.df_test['Promo'].value_counts(normalize=True) * 100

        # Print the numerical results
        logging.info("Training Promo Distribution (%):\n%s", train_promo_dist)
        logging.info("Test Promo Distribution (%):\n%s", test_promo_dist)

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        colors_train = ['#2c7f3d', '#d4af37']
        colors_test = ['#0b3954', '#ffba08']

        ax[0].bar(train_promo_dist.index, train_promo_dist.values, color=colors_train)
        ax[0].set_title('Promo Distribution in Training Set', fontsize=14, fontweight='bold')
        ax[0].set_xlabel('Promo', fontsize=12)
        ax[0].set_ylabel('Percentage (%)', fontsize=12)
        ax[0].set_ylim(0, 100)

        ax[1].bar(test_promo_dist.index, test_promo_dist.values, color=colors_test)
        ax[1].set_title('Promo Distribution in Test Set', fontsize=14, fontweight='bold')
        ax[1].set_xlabel('Promo', fontsize=12)
        ax[1].set_ylabel('Percentage (%)', fontsize=12)
        ax[1].set_ylim(0, 100)

        ax[0].grid(True, which='major', linestyle='--', linewidth=0.7)
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.7)

        plt.suptitle('Comparison of Promo Distribution Between Training and Test Sets', fontsize=16, fontweight='bold')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info("Promo distribution comparison plot saved to %s", save_path)

    def analyze_sales_around_holidays(self, save_path='../Images/sales_holidays_comparison.png'):
        logging.info("Analyzing sales around holidays...")
        self.df_train = self.df_train.sort_values('Date')
        holidays = self.df_train[self.df_train['StateHoliday'] != '0']['Date'].unique()

        before_sales = []
        during_sales = []
        after_sales = []

        for holiday in holidays:
            before_start = holiday - timedelta(days=7)
            before_end = holiday - timedelta(days=1)
            after_start = holiday + timedelta(days=1)
            after_end = holiday + timedelta(days=7)

            before_sales.extend(self.df_train[(self.df_train['Date'] >= before_start) & (self.df_train['Date'] <= before_end)]['Sales'])
            during_sales.extend(self.df_train[self.df_train['Date'] == holiday]['Sales'])
            after_sales.extend(self.df_train[(self.df_train['Date'] >= after_start) & (self.df_train['Date'] <= after_end)]['Sales'])

        avg_before = sum(before_sales) / len(before_sales) if before_sales else 0
        avg_during = sum(during_sales) / len(during_sales) if during_sales else 0
        avg_after = sum(after_sales) / len(after_sales) if after_sales else 0

        logging.info("Average Sales Before Holiday: %.2f", avg_before)
        logging.info("Average Sales During Holiday: %.2f", avg_during)
        logging.info("Average Sales After Holiday: %.2f", avg_after)

        comparison_df = pd.DataFrame({
            'Period': ['Before Holiday', 'During Holiday', 'After Holiday'],
            'Average Sales': [avg_before, avg_during, avg_after]
        })

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Period', y='Average Sales', data=comparison_df, palette='viridis')
        plt.title('Average Sales Before, During, and After Holidays', fontsize=16, fontweight='bold')
        plt.xlabel('Period', fontsize=14)
        plt.ylabel('Average Sales', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info("Sales analysis around holidays plot saved to %s", save_path)

    def analyze_holiday_sales(self, holiday_dates, window_days=7, save_path='../Images/holiday_sales_comparison.png'):
        logging.info("Analyzing holiday sales for specified dates...")
        
        def compute_average_sales(df, holiday_date, window_days):
            start_date = holiday_date - timedelta(days=window_days)
            end_date = holiday_date + timedelta(days=window_days)
            
            before_holiday = df[(df['Date'] >= start_date) & (df['Date'] < holiday_date)]
            during_holiday = df[(df['Date'] == holiday_date)]
            after_holiday = df[(df['Date'] > holiday_date) & (df['Date'] <= end_date)]
            
            avg_sales_before = before_holiday['Sales'].mean() if not before_holiday.empty else 0
            avg_sales_during = during_holiday['Sales'].mean() if not during_holiday.empty else 0
            avg_sales_after = after_holiday['Sales'].mean() if not after_holiday.empty else 0
            
            return avg_sales_before, avg_sales_during, avg_sales_after
        
        results = []
        for holiday in holiday_dates:
            avg_sales_before, avg_sales_during, avg_sales_after = compute_average_sales(self.df_train, pd.to_datetime(holiday), window_days)
            results.append({
                'Holiday': holiday,
                'Before Holiday': avg_sales_before,
                'During Holiday': avg_sales_during,
                'After Holiday': avg_sales_after
            })
            logging.info("Holiday: %s, Average Sales - Before: %.2f, During: %.2f, After: %.2f", holiday, avg_sales_before, avg_sales_during, avg_sales_after)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.melt(id_vars='Holiday', var_name='Period', value_name='Average Sales')

        sns.set_style("whitegrid")
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Holiday', y='Average Sales', hue='Period', data=results_df, palette='viridis')
        
        plt.title('Sales Behavior Around Holidays', fontsize=16, fontweight='bold')
        plt.xlabel('Holiday', fontsize=14)
        plt.ylabel('Average Sales', fontsize=14)
        plt.legend(title='Period')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info("Holiday sales analysis plot saved to %s", save_path)

    def analyze_customers_sales_correlation(self, save_path='../Images/customers_sales_correlation.png'):
        logging.info("Analyzing correlation between Customers and Sales...")
        if 'Customers' not in self.df_train.columns or 'Sales' not in self.df_train.columns:
            logging.error("Columns 'Customers' and 'Sales' must exist in the training dataset.")
            raise ValueError("Columns 'Customers' and 'Sales' must exist in the training dataset.")

        correlation = self.df_train[['Customers', 'Sales']].corr().loc['Customers', 'Sales']
        logging.info("Correlation between Customers and Sales: %.4f", correlation)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Customers', y='Sales', data=self.df_train, alpha=0.6)
        plt.title(f'Correlation between Customers and Sales: {correlation:.2f}', fontsize=16, fontweight='bold')
        plt.xlabel('Customers', fontsize=14)
        plt.ylabel('Sales', fontsize=14)
        plt.grid(True, linestyle='--', linewidth=0.7)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info("Customers vs Sales correlation plot saved to %s", save_path)

    def analyze_sales_trends(self, save_path='../Images/sales_trends.png'):
        logging.info("Analyzing sales trends over time...")
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=self.df_train, x='Date', y='Sales', estimator='mean', ci=None, color='blue')
        plt.title('Average Sales Trends Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Average Sales', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', linewidth=0.7)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info("Sales trends analysis plot saved to %s", save_path)
