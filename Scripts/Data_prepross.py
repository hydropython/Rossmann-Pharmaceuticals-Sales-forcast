import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import os

class Preprocessing:
    
    def __init__(self, train_file, test_file, submission_file, images_path='../Images/'):
        # Load the data
        self.df_train = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)
        self.df_submission = pd.read_csv(submission_file)

        # Merge df_test and df_submission on 'Id' to get 'Sales'
        self.df_test = pd.merge(self.df_test, self.df_submission, on='Id', how='left')

        # Create the images directory if it does not exist
        self.images_path = images_path
        os.makedirs(self.images_path, exist_ok=True)
        print(f"Data successfully loaded and merged. Images will be saved to {self.images_path}")
        
    def feature_engineering(self):
        """Extract features from date and other relevant information."""
        # Convert 'Date' column to datetime
        self.df_train['Date'] = pd.to_datetime(self.df_train['Date'])
        self.df_test['Date'] = pd.to_datetime(self.df_test['Date'])

        # Extract weekday and weekend features
        self.df_train['Weekday'] = self.df_train['Date'].dt.dayofweek
        self.df_train['Weekend'] = (self.df_train['Weekday'] >= 5).astype(int)

        self.df_test['Weekday'] = self.df_test['Date'].dt.dayofweek
        self.df_test['Weekend'] = (self.df_test['Weekday'] >= 5).astype(int)

        # Number of days to holidays (example: using a fixed holiday date)
        holidays = pd.to_datetime(['2021-12-25', '2022-01-01'])  # Add your holidays
        self.df_train['Days_to_Holiday'] = (holidays.min() - self.df_train['Date']).dt.days
        self.df_test['Days_to_Holiday'] = (holidays.min() - self.df_test['Date']).dt.days

        # Number of days after a holiday
        self.df_train['Days_After_Holiday'] = (self.df_train['Date'] - holidays.max()).dt.days
        self.df_test['Days_After_Holiday'] = (self.df_test['Date'] - holidays.max()).dt.days

        # Beginning of the month, mid-month, and end of the month
        self.df_train['Beginning_of_Month'] = (self.df_train['Date'].dt.day <= 10).astype(int)
        self.df_train['Mid_of_Month'] = ((self.df_train['Date'].dt.day > 10) & (self.df_train['Date'].dt.day <= 20)).astype(int)
        self.df_train['End_of_Month'] = (self.df_train['Date'].dt.day > 20).astype(int)

        self.df_test['Beginning_of_Month'] = (self.df_test['Date'].dt.day <= 10).astype(int)
        self.df_test['Mid_of_Month'] = ((self.df_test['Date'].dt.day > 10) & (self.df_test['Date'].dt.day <= 20)).astype(int)
        self.df_test['End_of_Month'] = (self.df_test['Date'].dt.day > 20).astype(int)

        # Additional features
        self.df_train['Is_Holiday'] = self.df_train['Date'].isin(holidays).astype(int)
        self.df_test['Is_Holiday'] = self.df_test['Date'].isin(holidays).astype(int)

        print("Feature engineering completed. Updated dataframes:")
        print(self.df_train.head())  # Display the first few rows of df_train
        print(self.df_test.head())    # Display the first few rows of df_test

    def handle_missing_values(self):
        # Fill missing values for 'Open' with mode since it's binary
        self.df_train['Open'].fillna(self.df_train['Open'].mode()[0], inplace=True)
        self.df_test['Open'].fillna(self.df_test['Open'].mode()[0], inplace=True)

        print("Missing values handled.")

    def encode_categorical(self):
        # Convert 'StateHoliday' to string type to ensure uniform encoding
        self.df_train['StateHoliday'] = self.df_train['StateHoliday'].astype(str)
        self.df_test['StateHoliday'] = self.df_test['StateHoliday'].astype(str)
        
        label_encoder = LabelEncoder()
        
        # Fit and transform categorical columns
        self.df_train['StateHoliday'] = label_encoder.fit_transform(self.df_train['StateHoliday'])
        self.df_test['StateHoliday'] = label_encoder.transform(self.df_test['StateHoliday'])
        
        print("Categorical variables encoded.")

    def scale_numeric_features(self):
        """Drop 'Customers' column from df_train and scale numeric features using StandardScaler."""
        # Check if 'Customers' column exists in df_train and drop it if found
        if 'Customers' in self.df_train.columns:
            self.df_train.drop(columns=['Customers'], inplace=True)
            print("'Customers' column dropped from df_train.")
        else:
            print("'Customers' column not found in df_train, skipping drop.")
        
        # Define numeric columns for scaling
        numeric_columns = ['Open', 'Promo', 'SchoolHoliday', 'DayOfWeek']  # Modify as per your dataset

        # Initialize the scaler
        scaler = StandardScaler()

        # Scale numeric columns in the training set
        self.df_train[numeric_columns] = scaler.fit_transform(self.df_train[numeric_columns])

        # Scale numeric columns in the test set
        self.df_test[numeric_columns] = scaler.transform(self.df_test[numeric_columns])

        print("Numeric features scaled.")
    def visualize_sales_distribution(self):
        # Visualize the sales distribution
        fig = px.histogram(self.df_train, x='Sales', nbins=50, title="Sales Distribution", color_discrete_sequence=['green'])
        
        # Save the plot
        fig.write_image(f"{self.images_path}/sales_distribution.png")
        print(f"Sales distribution plot saved at {self.images_path}/sales_distribution.png")
    
    def visualize_sales_over_time(self):
        # Convert 'Date' to datetime format if it's not already
        self.df_train['Date'] = pd.to_datetime(self.df_train['Date'])
        
        # Group by date and plot sales trend over time
        sales_over_time = self.df_train.groupby('Date')['Sales'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sales_over_time['Date'], y=sales_over_time['Sales'],
                                 mode='lines', name='Total Sales', line=dict(color='blue')))
        fig.update_layout(title='Sales Over Time', xaxis_title='Date', yaxis_title='Sales')

        # Save the plot
        fig.write_image(f"{self.images_path}/sales_over_time.png")
        print(f"Sales over time plot saved at {self.images_path}/sales_over_time.png")

    def scale_data(self):
        """Scale only the numeric columns in the training and test datasets."""
        # Initialize the scaler
        scaler = StandardScaler()

        # Select only numeric columns (excluding datetime and object columns)
        numeric_columns = self.df_train.select_dtypes(include=['float64', 'int64']).columns

        # Scale numeric columns in the training set
        X_train_scaled = scaler.fit_transform(self.df_train[numeric_columns])

        # Scale numeric columns in the test set
        X_test_scaled = scaler.transform(self.df_test[numeric_columns])

        print("Numeric features scaled successfully.")

        return X_train_scaled, X_test_scaled

    def preprocess(self):
        """Preprocess data and return scaled features and target variables."""
        
        # Assuming 'Sales' is the target column in the dataset
        target_column = 'Sales'  # Replace 'target_column' with 'Sales' (or the correct target name)
        
        # Drop the target column from features
        X_train = self.df_train.drop(columns=[target_column])
        y_train = self.df_train[target_column]
        
        X_val = self.df_test.drop(columns=[target_column])
        y_val = self.df_test[target_column]
        
        # Define numeric columns for scaling
        numeric_columns = ['Open', 'Promo', 'SchoolHoliday', 'DayOfWeek']  # Modify as per your dataset
        
        # Initialize the scaler
        scaler = StandardScaler()
        
        # Scale numeric columns in the training and validation sets
        X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
        X_val_scaled = scaler.transform(X_val[numeric_columns])
        
        print("Numeric features scaled.")
        
        return X_train_scaled, X_val_scaled, y_train, y_val  # Return 4 values