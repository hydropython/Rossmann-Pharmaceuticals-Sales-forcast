import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os

class EDA_Analysis:
    def __init__(self, train_file, test_file):
        # Load train and test datasets
        self.df_train = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)
       
        
        # Clean the Date column and convert to datetime
        self.df_train['Date'] = pd.to_datetime(self.df_train['Date'], errors='coerce')
        self.df_test['Date'] = pd.to_datetime(self.df_test['Date'], errors='coerce')
        
        # Drop rows where Date could not be converted
        self.df_train.dropna(subset=['Date'], inplace=True)
        self.df_test.dropna(subset=['Date'], inplace=True)
        
    def promo_distribution_comparison(self, save_path='../Images/promo_distribution_comparison.png'):
        """
        Compares the distribution of 'Promo' between training and test sets,
        prints the results, visualizes the comparison, and saves the plot.
        """
        # Check distribution of 'Promo' in both datasets
        train_promo_dist = self.df_train['Promo'].value_counts(normalize=True) * 100  # Percentage distribution
        test_promo_dist = self.df_test['Promo'].value_counts(normalize=True) * 100    # Percentage distribution

        # Print the numerical results
        print("Numerical Results:")
        print("Training Promo Distribution (%):\n", train_promo_dist)
        print("\nTest Promo Distribution (%):\n", test_promo_dist)

        # Set seaborn style for modern aesthetics
        sns.set_style("whitegrid")

        # Create a figure with subplots
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        # Define colors for modern visualization
        colors_train = ['#2c7f3d', '#d4af37']  # Dark green, Gold
        colors_test = ['#0b3954', '#ffba08']   # Dark blue, Bright yellow

        # Bar plot for training set
        ax[0].bar(train_promo_dist.index, train_promo_dist.values, color=colors_train)
        ax[0].set_title('Promo Distribution in Training Set', fontsize=14, fontweight='bold')
        ax[0].set_xlabel('Promo', fontsize=12)
        ax[0].set_ylabel('Percentage (%)', fontsize=12)
        ax[0].set_ylim(0, 100)

        # Bar plot for test set
        ax[1].bar(test_promo_dist.index, test_promo_dist.values, color=colors_test)
        ax[1].set_title('Promo Distribution in Test Set', fontsize=14, fontweight='bold')
        ax[1].set_xlabel('Promo', fontsize=12)
        ax[1].set_ylabel('Percentage (%)', fontsize=12)
        ax[1].set_ylim(0, 100)

        # Add gridlines and a clean seaborn look
        ax[0].grid(True, which='major', linestyle='--', linewidth=0.7)
        ax[1].grid(True, which='major', linestyle='--', linewidth=0.7)

        # Overall title
        plt.suptitle('Comparison of Promo Distribution Between Training and Test Sets', fontsize=16, fontweight='bold')

        # Adjust layout for better visual appeal
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot to the specified path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()

    def analyze_sales_around_holidays(self, save_path='../Images/sales_holidays_comparison.png'):
        """
        Analyzes and compares sales behavior before, during, and after holidays,
        prints the results, visualizes the comparison, and saves the plot.
        """
        # Ensure data is sorted by Date
        self.df_train = self.df_train.sort_values('Date')
        
        # Extract holidays and dates
        holidays = self.df_train[self.df_train['StateHoliday'] != '0']['Date'].unique()
        
        # Create lists to store sales data for each period
        before_sales = []
        during_sales = []
        after_sales = []
        
        for holiday in holidays:
            # Define time windows
            before_start = holiday - timedelta(days=7)
            before_end = holiday - timedelta(days=1)
            after_start = holiday + timedelta(days=1)
            after_end = holiday + timedelta(days=7)
            
            # Extract sales data
            before_sales.extend(self.df_train[(self.df_train['Date'] >= before_start) & (self.df_train['Date'] <= before_end)]['Sales'])
            during_sales.extend(self.df_train[self.df_train['Date'] == holiday]['Sales'])
            after_sales.extend(self.df_train[(self.df_train['Date'] >= after_start) & (self.df_train['Date'] <= after_end)]['Sales'])
        
        # Calculate average sales
        avg_before = sum(before_sales) / len(before_sales) if before_sales else 0
        avg_during = sum(during_sales) / len(during_sales) if during_sales else 0
        avg_after = sum(after_sales) / len(after_sales) if after_sales else 0
        
        # Print the numerical results
        print("Numerical Results:")
        print(f"Average Sales Before Holiday: {avg_before:.2f}")
        print(f"Average Sales During Holiday: {avg_during:.2f}")
        print(f"Average Sales After Holiday: {avg_after:.2f}")
        
        # Create a DataFrame for visualization
        comparison_df = pd.DataFrame({
            'Period': ['Before Holiday', 'During Holiday', 'After Holiday'],
            'Average Sales': [avg_before, avg_during, avg_after]
        })
        
        # Set seaborn style for modern aesthetics
        sns.set_style("whitegrid")
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Period', y='Average Sales', data=comparison_df, palette='viridis')
        plt.title('Average Sales Before, During, and After Holidays', fontsize=16, fontweight='bold')
        plt.xlabel('Period', fontsize=14)
        plt.ylabel('Average Sales', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        
        # Save the plot to the specified path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()
    def analyze_holiday_sales(self, holiday_dates, window_days=7, save_path='../Images/holiday_sales_comparison.png'):
        """
        Analyzes sales behavior around specified holidays.
        
        Parameters:
        holiday_dates (list): List of holiday dates in 'YYYY-MM-DD' format.
        window_days (int): Number of days before and after the holiday to analyze.
        save_path (str): Path to save the comparison plot.
        """
        # Define a function to compute average sales around each holiday
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
        
        # Collect results for each holiday
        results = []
        for holiday in holiday_dates:
            avg_sales_before, avg_sales_during, avg_sales_after = compute_average_sales(self.df_train, pd.to_datetime(holiday), window_days)
            results.append({
                'Holiday': holiday,
                'Before Holiday': avg_sales_before,
                'During Holiday': avg_sales_during,
                'After Holiday': avg_sales_after
            })
        
        # Convert results to DataFrame for plotting
        results_df = pd.DataFrame(results)
        results_df = results_df.melt(id_vars='Holiday', var_name='Period', value_name='Average Sales')

        # Print numerical results
        print("Numerical Results:")
        for result in results:
            print(f"Holiday: {result['Holiday']}")
            print(f"  Average Sales Before Holiday: {result['Before Holiday']:.2f}")
            print(f"  Average Sales During Holiday: {result['During Holiday']:.2f}")
            print(f"  Average Sales After Holiday: {result['After Holiday']:.2f}\n")
        
        # Set seaborn style for modern aesthetics
        sns.set_style("whitegrid")

        # Create bar plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Holiday', y='Average Sales', hue='Period', data=results_df, palette='viridis')
        
        # Titles and labels
        plt.title('Sales Behavior Around Holidays', fontsize=16, fontweight='bold')
        plt.xlabel('Holiday', fontsize=14)
        plt.ylabel('Average Sales', fontsize=14)
        plt.legend(title='Period')
        
        # Save and show plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    def analyze_customers_sales_correlation(self, save_path='../Images/customers_sales_correlation.png'):
        """
        Analyzes the correlation between 'Customers' and 'Sales' in the training dataset,
        prints the correlation coefficient, visualizes the correlation, and saves the plot.
        """
        # Check if 'Customers' and 'Sales' columns exist
        if 'Customers' not in self.df_train.columns or 'Sales' not in self.df_train.columns:
            raise ValueError("Columns 'Customers' and 'Sales' must exist in the training dataset.")

        # Compute the correlation between 'Customers' and 'Sales'
        correlation = self.df_train[['Customers', 'Sales']].corr().loc['Customers', 'Sales']

        # Print the correlation result
        print(f"Numerical Result: Correlation between Customers and Sales: {correlation:.4f}")

        # Set seaborn style for modern aesthetics
        sns.set_style("whitegrid")

        # Create a scatter plot with regression line
        plt.figure(figsize=(10, 6))
        sns.regplot(x='Customers', y='Sales', data=self.df_train, scatter_kws={'color':'#2c7f3d'}, line_kws={'color':'#d4af37'})
        
        # Title and labels
        plt.title('Correlation between Customers and Sales', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Customers', fontsize=14)
        plt.ylabel('Sales', fontsize=14)
        
        # Save and show plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

