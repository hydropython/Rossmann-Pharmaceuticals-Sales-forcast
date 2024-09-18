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