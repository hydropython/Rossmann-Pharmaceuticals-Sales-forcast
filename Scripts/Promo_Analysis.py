import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Promo_Analysis:
    def __init__(self, df_train, df_test, df_store, df_sample_submission):
        # Initialize data
        self.df_train = df_train
        self.df_test = df_test
        self.df_store = df_store
        self.df_sample_submission = df_sample_submission
        
        # Merge df_test with df_sample_submission on 'Id'
        self.df_test = pd.merge(self.df_test, self.df_sample_submission, on='Id', how='left')
        
        # Merge df_train with df_store on 'Store'
        self.df_train = pd.merge(self.df_train, self.df_store, left_on='Store', right_on='Store', how='left')

    def analyze_promo_sales_impact(self, save_path='../Images/promo_sales_impact.png'):
        """
        Analyzes the impact of promotions on sales and customer counts using a pie chart.
        """
        # Calculate average sales with and without promotions
        promo_sales = self.df_train[self.df_train['Promo'] == 1]['Sales']
        no_promo_sales = self.df_train[self.df_train['Promo'] == 0]['Sales']
        
        avg_promo_sales = promo_sales.mean() if not promo_sales.empty else 0
        avg_no_promo_sales = no_promo_sales.mean() if not no_promo_sales.empty else 0
        
        # Print results
        print(f"Average Sales with Promo: {avg_promo_sales:.2f}")
        print(f"Average Sales without Promo: {avg_no_promo_sales:.2f}")
        
        # Ensure the Images folder exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Set seaborn style for modern aesthetics
        sns.set_style("whitegrid")
        
        # Create a DataFrame for visualization
        sales_comparison_df = pd.DataFrame({
            'Promo Status': ['With Promo', 'Without Promo'],
            'Average Sales': [avg_promo_sales, avg_no_promo_sales]
        })
        
        # Create a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(
            sales_comparison_df['Average Sales'],
            labels=sales_comparison_df['Promo Status'],
            autopct='%1.1f%%',
            colors=['#1f77b4', '#ff7f0e'],
            startangle=140,
            wedgeprops=dict(width=0.4)
        )
        plt.title('Impact of Promotions on Sales', fontsize=16, fontweight='bold')
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def recommend_promo_deployment(self, save_path='../Images/recommend_promo_deployment.png'):
        """
        Recommends the most effective way to deploy promotions based on store performance.
        """
        # Calculate average sales per store with and without promotions
        promo_sales_store = self.df_train[self.df_train['Promo'] == 1].groupby('Store')['Sales'].mean()
        no_promo_sales_store = self.df_train[self.df_train['Promo'] == 0].groupby('Store')['Sales'].mean()
        
        # Merge the results into a single DataFrame for comparison
        store_sales_comparison = pd.DataFrame({
            'Average Sales with Promo': promo_sales_store,
            'Average Sales without Promo': no_promo_sales_store
        }).fillna(0)
        
        # Calculate the difference to recommend stores for promotion
        store_sales_comparison['Sales Increase'] = store_sales_comparison['Average Sales with Promo'] - store_sales_comparison['Average Sales without Promo']
        
        # Sort stores by sales increase
        store_sales_comparison = store_sales_comparison.sort_values(by='Sales Increase', ascending=False)
        
        # Print the top stores for promotion
        print("Stores recommended for promotions based on sales increase:")
        print(store_sales_comparison.head())
        
        # Ensure the Images folder exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Set seaborn style for modern aesthetics
        sns.set_style("whitegrid")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=store_sales_comparison.index, y='Sales Increase', data=store_sales_comparison, palette='viridis')
        plt.title('Recommended Stores for Promotions Based on Sales Increase', fontsize=16, fontweight='bold')
        plt.xlabel('Store', fontsize=14)
        plt.ylabel('Sales Increase', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()