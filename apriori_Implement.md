# Apriori Algorithm Implementation using Jupyter Labs

This notebook demonstrates the implementation of the **Apriori algorithm** for association rule mining in **Jupyter Labs**.

## Objective:
- We have used a market basket dataset of 9825 transactions including 169 unique items for this demonstration  **grocery basket**.
- *Source* :https://www.kaggle.com/datasets/irfanasrullah/groceries



```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import time
import sys
```

**Calculate runtime and memory usage**

This function when called calculates each function runtime and memory usage and will post the data at the end for each function of the code



```python
# Function to calculate runtime and memory usage
def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = sys.getsizeof(args) + sys.getsizeof(kwargs)

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = sys.getsizeof(result)

        runtime = end_time - start_time
        memory_used = end_memory - start_memory

        print(f"Function {func.__name__} took {runtime:.4f} seconds and used {memory_used} bytes of memory.")
        return result
    return wrapper

```

**Function to pre-process the dataset on which the apriori algorithm will be performed**
- Group the dataset by column as in this case there are three columns Member_number, Date, itemDescription hence we are classifying groups for each of these columns.
- Here, we have performed one-hot encoding on the dataset.One-hot encoding is necessary for the Apriori algorithm because the algorithm operates on binary transactions, where each item is represented as either present (1) or absent (0) in a transaction. Since Apriori is a frequent itemset mining algorithm, it requires data in a structured format that clearly indicates whether an item appears in each transaction.


```python
@measure_performance
def load_and_preprocess_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return None

    # Group by Member number and aggregate Item details into lists
    transactions = df.groupby('Member_number')['itemDescription'].apply(list).tolist()

    # One-hot encode the transactions
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    return df_encoded
```

**Function to perform Apriori and association rule mining on the dataset**


```python
@measure_performance
def apply_apriori(df_encoded, min_support=0.01):
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True, low_memory=True)  # low_memory=True
    return frequent_itemsets


@measure_performance
def generate_association_rules(frequent_itemsets, min_confidence=0.01, min_lift=1.0):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules_with_lift = rules[rules['lift'] > min_lift]
    return rules_with_lift
```

**Print and load the dataset**


```python
csv_file = "groceries_dataset.csv" 

df_encoded = load_and_preprocess_data(csv_file)

if df_encoded is None:
    exit()
```

    Function load_and_preprocess_data took 0.4720 seconds and used 651018 bytes of memory.


**Set the minimum thresholds for association rule mining**
-This section calculates the confidence, support and lift parameters for the entire dataset 



```python
min_supports_to_try = [0.005, 0.002, 0.001]  # Lower support for large datasets
min_confidences_to_try = [0.01, 0.005, 0.002]
min_lift = 1.0

found_rules = False

for min_support in min_supports_to_try:
    print(f"Trying min_support = {min_support}")
    frequent_itemsets = apply_apriori(df_encoded, min_support)

    print("\nFrequent Itemsets (Top 50 or all if fewer):")
    print(frequent_itemsets.head(min(50, len(frequent_itemsets))))

    for min_confidence in min_confidences_to_try:
        print(f"Trying min_confidence = {min_confidence}")
        rules = generate_association_rules(frequent_itemsets, min_confidence, min_lift)

        if not rules.empty:
            found_rules = True
            print(f"\nAssociation Rules (support={min_support}, confidence={min_confidence}, lift > {min_lift}):")
            print(rules[['antecedents', 'consequents', 'confidence', 'lift', 'support']].head(min(50, len(rules))))
            break

    if found_rules:
        break

if not found_rules:
    print("No association rules found. Check your data, format, or if association rule mining is appropriate.")
```

    Trying min_support = 0.005
    Function apply_apriori took 0.2954 seconds and used 2408524 bytes of memory.
    
    Frequent Itemsets (Top 50 or all if fewer):
         support                    itemsets
    0   0.015393     (Instant food products)
    1   0.078502                  (UHT-milk)
    2   0.005644          (abrasive cleaner)
    3   0.007440          (artif. sweetener)
    4   0.031042             (baking powder)
    5   0.119548                      (beef)
    6   0.079785                   (berries)
    7   0.062083                 (beverages)
    8   0.158799              (bottled beer)
    9   0.213699             (bottled water)
    10  0.009749                    (brandy)
    11  0.135967               (brown bread)
    12  0.126475                    (butter)
    13  0.064905               (butter milk)
    14  0.022832                  (cake bar)
    15  0.016932                   (candles)
    16  0.053874                     (candy)
    17  0.165213               (canned beer)
    18  0.029502               (canned fish)
    19  0.005387              (canned fruit)
    20  0.020523         (canned vegetables)
    21  0.043869                  (cat food)
    22  0.010775                   (cereals)
    23  0.044638               (chewing gum)
    24  0.100564                   (chicken)
    25  0.086455                 (chocolate)
    26  0.015393     (chocolate marshmallow)
    27  0.185480              (citrus fruit)
    28  0.007953                   (cleaner)
    29  0.018728           (cling film/bags)
    30  0.114931                    (coffee)
    31  0.023858            (condensed milk)
    32  0.088507             (cream cheese )
    33  0.120831                      (curd)
    34  0.011801               (curd cheese)
    35  0.008466               (dental care)
    36  0.086455                   (dessert)
    37  0.032581                 (detergent)
    38  0.018728              (dish cleaner)
    39  0.034120                    (dishes)
    40  0.017188                  (dog food)
    41  0.133145             (domestic eggs)
    42  0.010262  (female sanitary products)
    43  0.016419         (finished products)
    44  0.007440                      (fish)
    45  0.036429                     (flour)
    46  0.017188            (flower (seeds))
    47  0.137506               (frankfurter)
    48  0.023089            (frozen dessert)
    49  0.025911               (frozen fish)
    Trying min_confidence = 0.01
    Function generate_association_rules took 0.1212 seconds and used 40510704 bytes of memory.
    
    Association Rules (support=0.005, confidence=0.01, lift > 1.0):
                    antecedents              consequents  confidence      lift  \
    0              (rolls/buns)  (Instant food products)    0.015407  1.000954   
    1   (Instant food products)             (rolls/buns)    0.350000  1.000954   
    2   (Instant food products)        (root vegetables)    0.450000  1.951168   
    3         (root vegetables)  (Instant food products)    0.030033  1.951168   
    4                    (soda)  (Instant food products)    0.025368  1.648091   
    5   (Instant food products)                   (soda)    0.516667  1.648091   
    6   (Instant food products)             (whole milk)    0.516667  1.127641   
    7              (whole milk)  (Instant food products)    0.017357  1.127641   
    8                (UHT-milk)                   (beef)    0.133987  1.120775   
    9                    (beef)               (UHT-milk)    0.087983  1.120775   
    10                (berries)               (UHT-milk)    0.093248  1.187840   
    11               (UHT-milk)                (berries)    0.094771  1.187840   
    12              (beverages)               (UHT-milk)    0.099174  1.263328   
    13               (UHT-milk)              (beverages)    0.078431  1.263328   
    14           (bottled beer)               (UHT-milk)    0.093700  1.193597   
    15               (UHT-milk)           (bottled beer)    0.189542  1.193597   
    16          (bottled water)               (UHT-milk)    0.099640  1.269268   
    17               (UHT-milk)          (bottled water)    0.271242  1.269268   
    18            (brown bread)               (UHT-milk)    0.090566  1.153681   
    19               (UHT-milk)            (brown bread)    0.156863  1.153681   
    20                 (butter)               (UHT-milk)    0.083164  1.059394   
    21               (UHT-milk)                 (butter)    0.133987  1.059394   
    22            (butter milk)               (UHT-milk)    0.102767  1.309101   
    23               (UHT-milk)            (butter milk)    0.084967  1.309101   
    24                  (candy)               (UHT-milk)    0.138095  1.759135   
    25               (UHT-milk)                  (candy)    0.094771  1.759135   
    26            (canned beer)               (UHT-milk)    0.090062  1.147262   
    27               (UHT-milk)            (canned beer)    0.189542  1.147262   
    28               (UHT-milk)                (chicken)    0.133987  1.332350   
    29                (chicken)               (UHT-milk)    0.104592  1.332350   
    30               (UHT-milk)              (chocolate)    0.114379  1.322996   
    31              (chocolate)               (UHT-milk)    0.103858  1.322996   
    34               (UHT-milk)                 (coffee)    0.163399  1.421715   
    35                 (coffee)               (UHT-milk)    0.111607  1.421715   
    36          (cream cheese )               (UHT-milk)    0.118841  1.513858   
    37               (UHT-milk)          (cream cheese )    0.133987  1.513858   
    38                   (curd)               (UHT-milk)    0.112527  1.433426   
    39               (UHT-milk)                   (curd)    0.173203  1.433426   
    40                (dessert)               (UHT-milk)    0.080119  1.020597   
    41               (UHT-milk)                (dessert)    0.088235  1.020597   
    42          (domestic eggs)               (UHT-milk)    0.098266  1.251766   
    43               (UHT-milk)          (domestic eggs)    0.166667  1.251766   
    44               (UHT-milk)            (frankfurter)    0.192810  1.402192   
    45            (frankfurter)               (UHT-milk)    0.110075  1.402192   
    46               (UHT-milk)           (frozen meals)    0.081699  1.299853   
    47           (frozen meals)               (UHT-milk)    0.102041  1.299853   
    48      (frozen vegetables)               (UHT-milk)    0.110000  1.401242   
    49               (UHT-milk)      (frozen vegetables)    0.143791  1.401242   
    50               (UHT-milk)  (fruit/vegetable juice)    0.143791  1.150917   
    51  (fruit/vegetable juice)               (UHT-milk)    0.090349  1.150917   
    
         support  
    0   0.005387  
    1   0.005387  
    2   0.006927  
    3   0.006927  
    4   0.007953  
    5   0.007953  
    6   0.007953  
    7   0.007953  
    8   0.010518  
    9   0.010518  
    10  0.007440  
    11  0.007440  
    12  0.006157  
    13  0.006157  
    14  0.014879  
    15  0.014879  
    16  0.021293  
    17  0.021293  
    18  0.012314  
    19  0.012314  
    20  0.010518  
    21  0.010518  
    22  0.006670  
    23  0.006670  
    24  0.007440  
    25  0.007440  
    26  0.014879  
    27  0.014879  
    28  0.010518  
    29  0.010518  
    30  0.008979  
    31  0.008979  
    34  0.012827  
    35  0.012827  
    36  0.010518  
    37  0.010518  
    38  0.013597  
    39  0.013597  
    40  0.006927  
    41  0.006927  
    42  0.013084  
    43  0.013084  
    44  0.015136  
    45  0.015136  
    46  0.006414  
    47  0.006414  
    48  0.011288  
    49  0.011288  
    50  0.011288  
    51  0.011288  



```python

```
