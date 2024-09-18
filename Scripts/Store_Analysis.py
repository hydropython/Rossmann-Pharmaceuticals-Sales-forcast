import pandas as pd

class StoreAnalysis:
    def __init__(self, df_combined):
        self.df_combined = df_combined

    def stores_open_all_weekdays(self):
        self.df_combined['DayOfWeek'] = self.df_combined['DayOfWeek'].astype(str)
        open_weekdays = self.df_combined.groupby('Store')['DayOfWeek'].apply(lambda x: set(x) == {'1', '2', '3', '4', '5', '6', '7'})
        stores_open_weekdays = open_weekdays[open_weekdays].index
        df_weekend_sales = self.df_combined[self.df_combined['Store'].isin(stores_open_weekdays)]
        df_weekend_sales = df_weekend_sales[df_weekend_sales['DayOfWeek'].isin(['6', '7'])]
        avg_weekend_sales = df_weekend_sales.groupby('Store')['Sales'].mean()
        return avg_weekend_sales

    def assortment_impact_on_sales(self):
        assortment_sales = self.df_combined.groupby('Assortment')['Sales'].mean()
        return assortment_sales

    def competition_distance_impact(self):
        df_distance_sales = self.df_combined.dropna(subset=['CompetitionDistance'])
        distance_sales = df_distance_sales.groupby('CompetitionDistance')['Sales'].mean()
        return distance_sales

    
    def competition_distance_city_center(self):
        # Placeholder: Example of how you might define `distance_sales`
        # Ensure that distance_sales is defined before using it
        distance_sales = pd.DataFrame({
            'CompetitionDistance': [1000, 2000, 3000],
            'Sales': [200, 300, 400]
        })
        
        # Filter for city center stores
        city_center_stores = self.df_combined[self.df_combined['StoreType'] == 'c']
        
        # Merge with distance_sales on CompetitionDistance
        df_city_center_distance = pd.merge(city_center_stores, distance_sales, on='CompetitionDistance', how='left')
        
        return df_city_center_distance

    def competition_opening_impact(self):
        df_initial_na = self.df_combined[self.df_combined['CompetitionDistance'].isna()]
        df_later_values = self.df_combined[self.df_combined['CompetitionDistance'].notna()]
        stores_with_changes = set(df_initial_na['Store']).intersection(set(df_later_values['Store']))
        df_stores_with_changes = self.df_combined[self.df_combined['Store'].isin(stores_with_changes)]
        return df_stores_with_changes