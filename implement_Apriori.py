import pandas as pd
import numpy as np
import random
from mlxtend.frequent_patterns import apriori, association_rules

# Updated list of commonly purchased items
items = [
    'Apples', 'Bananas', 'Oranges', 'Grapes', 'Tomatoes', 'Lettuce', 'Carrots', 'Potatoes', 'Cucumbers', 'Onions',
    'Broccoli', 'Bell Peppers', 'Spinach', 'Zucchini', 'Eggplant', 'Cabbage', 'Kale', 'Cauliflower', 'Mushrooms',
    'Avocados', 'Lemons', 'Limes', 'Garlic', 'Fresh Herbs', 'Baby Diapers', 'Baby Wipes', 'Baby Lotion', 'Baby Shampoo',
    'Baby Powder', 'Baby Clothes', 'Baby Bottles', 'Baby Formula', 'Baby Snacks', 'Baby Rash Cream', 'Baby Carrier',
    'Diaper Bag', 'Baby Crib', 'Baby High Chair', 'Baby Stroller', 'Baby Teething Rings', 'Lipstick', 'Lip Gloss',
    'Lip Balm', 'Face Powder', 'Blush', 'Mascara', 'Eyeliner', 'Foundation', 'Concealer', 'Nail Polish',
    'Makeup Brushes',
    'Setting Spray', 'Face Cream', 'Face Masks', 'Hand Cream', 'Hair Shampoo', 'Hair Conditioner', 'Hair Gel',
    'Hair Oil',
    'Hairbrush', 'Facial Cleanser', 'Makeup Remover', 'Cotton Swabs', 'Toothpaste', 'Toothbrush', 'Mouthwash',
    'Deodorant',
    'Body Wash', 'Body Lotion', 'Razor', 'Shaving Cream', 'Bath Towels', 'Shower Curtain', 'Bath Mat', 'Loofah',
    'Laundry Detergent', 'Fabric Softener', 'Ironing Board', 'Iron', 'Vacuum Cleaner', 'Broom', 'Dustpan', 'Mop',
    'Toilet Paper', 'Paper Towels', 'Trash Bags', 'Dish Soap', 'Dishwashing Sponges', 'Cutting Boards', 'Knives',
    'Pots and Pans', 'Cooking Utensils', 'Mixing Bowls', 'Bakeware', 'Can Opener', 'Tupperware', 'Coffee Maker',
    'Toaster', 'Blender', 'Plates, Bowls, and Cups'
]


# Generate a synthetic grocery store transaction dataset with customer transaction IDs
def generate_grocery_dataset(num_transactions=1000, max_transaction_size=20):
    transactions = []

    for transaction_id in range(1, num_transactions + 1):
        # Reduce the maximum transaction size to avoid large transactions
        transaction_size = np.random.randint(1, max_transaction_size + 1)
        transaction = random.sample(items, transaction_size)
        transaction_data = [1 if item in transaction else 0 for item in items]
        transactions.append([transaction_id] + transaction_data)

    df = pd.DataFrame(transactions, columns=['Transaction_ID'] + items)

    return df


# Generate grocery store dataset
df = generate_grocery_dataset(num_transactions=500, max_transaction_size=15)

# Print the generated random dataset with Transaction IDs
print("Random Grocery Store Transactions:")
print(df.head())

# Reset index before applying Apriori so 'Transaction_ID' is a column
df_apriori = df.reset_index(drop=True)

# Apply Apriori algorithm (excluding 'Transaction_ID' column)
df_apriori_items = df_apriori.drop(columns=['Transaction_ID'])

# Apply apriori algorithm with a lower min_support (0.02) to increase the chances of finding frequent itemsets
frequent_itemsets = apriori(df_apriori_items, min_support=0.02, use_colnames=True)

# Generate association rules with a lower confidence threshold (0.05)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)

# Calculate lift manually (lift = confidence / antecedent support)
rules['lift'] = rules['confidence'] / rules['antecedent support']

# Display results
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules with Confidence and Lift:")
print(rules[['antecedents', 'consequents', 'confidence', 'lift']])
