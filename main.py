import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def analyze_superstore_purchases(data_file, min_support=0.01, min_confidence=0.2):
    """
    Analyzes superstore purchase invoices using association rule mining (Apriori).

    Args:
        data_file (str): Path to the CSV file containing customer purchase data.
                           The CSV should have at least two columns: 'InvoiceNo' and 'StockCode' (or similar).
        min_support (float): Minimum support threshold for frequent itemsets.
        min_confidence (float): Minimum confidence threshold for association rules.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated association rules, or None if an error occurs.
                           Returns an empty DataFrame if no rules meet the criteria.
    """

    try:
        # 1. Load the data
        df = pd.read_csv(data_file)

        # 2. Data preprocessing (crucial for Apriori)
        # Ensure proper data types and handle potential missing values.
        # Convert 'InvoiceNo' to string to handle mixed data types.
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)
        df.dropna(inplace=True)  # Remove rows with any missing values

        # Group by InvoiceNo and create a list of items purchased in each transaction
        basket = df.groupby('InvoiceNo')['StockCode'].apply(list).reset_index()

        # One-hot encode the items for Apriori
        def encode_units(x):
            if x <= 0:
                return 0
            if x >= 1:
                return 1

        basket_sets = basket.groupby('InvoiceNo')['StockCode'].value_counts().unstack(fill_value=0).applymap(
            encode_units)

        # 3. Apply Apriori algorithm
        frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

        # 4. Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        # 5. Sort rules for better readability (optional)
        rules.sort_values('confidence', ascending=False, inplace=True)

        return rules

    except FileNotFoundError:
        print(f"Error: File not found at {data_file}")
        return None
    except KeyError:
        print("Error: The CSV file must contain 'InvoiceNo' and 'StockCode' (or similar) columns.")
        return None
    except Exception as e:  # Catch other potential errors during processing
        print(f"An error occurred: {e}")
        return None


# Example usage (replace 'your_data.csv' with the actual path to your data file):
file_path = 'your_data.csv'  # Update with your file path.
rules_df = analyze_superstore_purchases(file_path, min_support=0.02, min_confidence=0.3)  # Adjust thresholds as needed

if rules_df is not None:
    if not rules_df.empty:
        print(rules_df.head())  # Display the top rules
        # You can further analyze and filter the rules_df as needed.
        # For example, to find rules where 'Product A' leads to 'Product B':
        # print(rules_df[ (rules_df['antecedents'].apply(lambda x: 'Product A' in x)) & (rules_df['consequents'].apply(lambda x: 'Product B' in x)) ])

    else:
        print("No association rules found that meet the specified criteria.")


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
