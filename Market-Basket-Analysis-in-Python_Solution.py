#!/usr/bin/env python
# coding: utf-8

# ## **Market Basket Analysis in Python**
# 
# In this exercise, you will learn how to:
# * Identify patterns in consumer decision-making with the `mlxtend` package.
# * Use metrics to evaluate the properties of patterns.
# * Construct "rules" that provide concrete recommendations for businesses.
# * Visualize patterns and rules using `seaborn` and `matplotlib`.
# 
# ## **The dataset**
# 
# **We'll use a dataset from a Brazilian ecommerce site (olist.com) that is divided into three CSV files:**
# 
# 1. `olist_order_items_dataset.csv`
# 2. `olist_products_dataset.csv`
# 3. `product_category_name_translation.csv`
# 
# **The column definitions are as follows:**
# 
# `olist_order_items_dataset.csv`:
# 
# - `order_id`: The unique identifier for a transaction.
# - `order_item_id`: The order of an item within a transaction.
# - `product_id`: The unique identifier for a product.
# - `price`: The product's price.
# 
# `olist_products_dataset.csv`:
# 
# - `product_id`: The unique identifier for a product.
# - `product_category_name`: The name of an item's product category in Portuguese.
# - `product_weight_g`: The product's weight in grams.
# - `product_length_cm`: The product's length in centimeters.
# - `product_width_cm`: The product's width in centimeters.
# - `product_height_cm`: The product's height in centimeters.
# 
# `product_category_name_translation.csv`:
# 
# - `product_category_name`: The name of an item's product category in Portuguese.
# - `product_category_name_english`: The name of an item's product category in English.
# 

# ## **Data preparation**

# The first step in any Market Basket Analysis (MBA) project is to determine what constitutes an **item**, an **itemset**, and a **transaction**. This will depend on the dataset we're using and the question we're attempting to answer.
# 
# * **Grocery store**
# 	* Item: Grocery
# 	* Itemset: Collection of groceries
# 	* Transaction: Basket of items purchased
# * **Music streaming service**
# 	* Item: Song
# 	* Itemset: Collection of unique songs
# 	* Transaction: User song library
# * **Ebook store**
# 	* Item: Ebook
# 	* Itemset: One or more ebooks
# 	* Transaction: User ebook library
# 

# **In this exercise, we'll use a dataset of transactions from olist.com, a Brazilian ecommerce site.**
# * 100,000+ orders over 2016-2018.
# * Olist connects sellers to marketplaces.
# * Seller can register products with Olist.
# * Customer makes purchase at marketplace from Olist store.
# * Seller fulfills orders.

# **What is an item**?
#   * A product purchased from Olist.
# 
# **What is an itemset?**
#   * A collection of one or more product(s).
# 
# **What is a transaction?**
#   * An itemset that corresponds to a customer's order.

# In[1]:


# Import modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set default asthetic parameters.
sns.set()

# Define path to data.
data_path = 'https://github.com/datacamp/Market-Basket-Analysis-in-python-live-training/raw/master/data/'


# In[2]:


# Load orders dataset.
orders = pd.read_csv(data_path+'olist_order_items_dataset.csv')

# Load products items dataset.
products = pd.read_csv(data_path+'olist_products_dataset.csv')

# Load translations dataset.
translations = pd.read_csv(data_path+'product_category_name_translation.csv')


# In[3]:


# Print orders header.
orders.head()


# In[4]:


# Print orders info.
orders.info()


# In[5]:


# Print products header.
products.head()


# In[6]:


# Print products info.
products.info()


# In[7]:


# Print translations header.
translations.head()


# In[8]:


# Print translations info.
translations.info()


# ---
# <center><h1> Q&A 1</h1> </center>
# 
# ---

# ### **Translating item category names**

# **The product names are given in Portuguese.**
#   * We'll translate the names to English using a `pandas` `DataFrame` named `translations`.
#   * `.merge()` performs a join operation on columns or indices.
#   * `on` is the column on which to perform the join.
#   * `how` specifies which keys to use to perform the join. 

# In[9]:


# Translate product names to English.
products = products.merge(translations, on='product_category_name', how="left")

# Print English names.
products['product_category_name_english']


# ### **Convert product IDs to product category names.**

# **We can work with product IDs directly, but do not have product names.**
#   * Map product IDs to product category names, which are available in `products`.
#   * Use another `.merge()` with `orders` and subset of `products` columns.
#   
# **Using category names will also simplify the analysis, since there are fewer categories than products.**

# In[10]:


# Define product category name in orders DataFrame.
orders = orders.merge(products[['product_id','product_category_name_english']], on='product_id', how='left')


# In[11]:


# Print orders header.
orders.head()


# In[12]:


# Drop products without a defined category.
orders.dropna(inplace=True, subset=['product_category_name_english'])


# In[13]:


# Print number of unique items.
len(orders['product_id'].unique())


# In[14]:


# Print number of unique categories.
len(orders['product_category_name_english'].unique())


# **Insight**: Performing "aggregation" up to the product category level reduces the number of potential itemsets from $2^{32328}$ to $2^{71}$.

# ### **Construct transactions from order and product data**

# * **We will perform Market Basket Analysis on transactions.**
#   * A transaction consists of the unique items purchased by a customer.
# * **Need to extract transactions from orders `DataFrame`.**
#   * Group all items in an order.

# In[15]:


# Identify transactions associated with example order.
example1 = orders[orders['order_id'] == 'fe64170e936bc5f6a6a41def260984b9']['product_category_name_english']

# Print example.
example1


# In[16]:


# Identify transactions associated with example order.
example2 = orders[orders['order_id'] == 'fffb9224b6fc7c43ebb0904318b10b5f']['product_category_name_english']

# Print example.
example2


# **Insight**: Aggregation reduces the number of items and, therefore, itemsets.

# **Map `orders` to `transactions`.**
# * `.groupby()` splits a `DataFrame` into groups according to some criterion.
# * `.unique()` returns list of unique values.

# In[17]:


# Recover transaction itemsets from orders DataFrame.
transactions = orders.groupby("order_id").product_category_name_english.unique()

# Print transactions header.
transactions.head()


# In[18]:


# Plot 50 largest categories of transactions.
transactions.value_counts()[:50].plot(kind='bar', figsize=(15,5))


# **Insight 1:** The most common itemsets consist of a single item.
# 
# **Insight 2:** There's a long tail of categories that consist of infrequently purchased items.

# **Use `.tolist()` to transform a `DataFrame` or `Series` object into a list.**

# In[19]:


# Convert the pandas series to list of lists.
transactions = transactions.tolist()


# ### **Summarize final transaction data**

# In[20]:


# Print length of transactions.
len(transactions)


# In[21]:


# Count number of unique item categories for each transaction.
counts = [len(transaction) for transaction in transactions]


# In[22]:


# Print median number of items in a transaction.
np.median(counts)


# In[23]:


# Print maximum number of items in a transaction.
np.max(counts)


# ---
# <center><h1> Q&A 2</h1> </center>
# 
# ---

# ## **Association Rules and Metrics**

# **Association rule:** an "if-then" relationship between two itemsets.
#   * **rule:** if *{coffee)* then *{milk}*.
#   * **antecedent:** coffee
#   * **consequent:** milk
# 
# **Metric:** a measure of the strength of association between two itemsets.
#   * **rule:** if *{coffee)* then *{milk}*
#   * **support:** 0.10
#   * **leverage:** 0.03
# 
# 

# ### **One-hot encode the transaction data**

# * **One-hot encoding data.**
#   * `TransactionEncoder()` instantiates an encoder object.
#   * `.fit()` creates mapping between list and one-hot encoding.
#   * `.transform()` transforms list into one-hot encoded array.

# * **Applying one-hot encoding will transform the list of lists (of transactions) into a `DataFrame`.**
#   * The columns correspond to item categories and the rows correspond to transactions. A true indicates that a transaction contains an item from the corresponding category.
# * **One-hot encoding simplifies the computation of metrics.**
#   * We will also use a one-hot encoded `DataFrame` as an input to different `mlxtend` functions.

# In[25]:


pip install mlxtend


# In[26]:


from mlxtend.preprocessing import TransactionEncoder

# Instantiate an encoder.
encoder = TransactionEncoder()

# Fit encoder to list of lists.
encoder.fit(transactions)

# Transform lists into one-hot encoded array.
onehot = encoder.transform(transactions)

# Convert array to pandas DataFrame.
onehot = pd.DataFrame(onehot, columns = encoder.columns_)


# In[27]:


# Print header.
onehot.head()


# 
# ### **Compute the support metric**
# 
# * Support measures the frequency with which an itemset appears in a database of transactions.

# 
# $$support(X) = \frac{\text{number of transactions containing X}}{\text{total number of transactions}}$$

# * `.mean(axis=0)` computes support values for one-hot encoded `DataFrame`.  
# * A high support value indicates that items in an itemset are purchased together frequently and, thus, are associated with each other.

# In[28]:


# Print support metric over all rows for each column.
onehot.mean(axis=0)


# **Observation:** In retail and ecommerce settings, any particular item is likely to account for a small share of transactions. Here, we've aggregated up to the product category level and very popular categories are still only present in 5% of transactions. Consequently, itemsets with 2 or more item categories will account for a vanishingly small share of total transactions (e.g. 0.01%).

# ### **Compute the item count distribution over transactions**

# * `onehot.sum(axis=1)` sums across the columns in a `DataFrame`. 

# In[29]:


# Print distribution of item counts.
onehot.sum(axis=1).value_counts()


# **Insight:** Only 726 transactions contain more than one item category. We may want to consider whether aggregation discards too many multi-item itemsets.

# ### **Create a column for an itemset with multiple items**

# * **We can create multi-item columns using the logical AND operation.**
#   * `True & True = True`
#   * `True & False = False`
#   * `False & True = False`
#   * `False & False = False`

# In[30]:


# Add sports_leisure and health_beauty to DataFrame.
onehot['sports_leisure_health_beauty'] = onehot['sports_leisure'] & onehot['health_beauty']

# Print support value.
onehot['sports_leisure_health_beauty'].mean(axis = 0)


# **Insight:** Only 0.014% of transactions contain a product from both the sports and leisure, and health and beauty categories. These are typically the type of numbers we will work with when we set pruning thresholds in the following section.

# ### **Aggregate the dataset further by combining product sub-categories**

# * **We can use the inclusive OR operation to combine multiple categories.**
#   * `True | True = True`
#   * `True | False = True`
#   * `False | True = True`
#   * `False | False = False`

# In[31]:


# Merge books_imported and books_technical.
onehot['books'] = onehot['books_imported'] | onehot['books_technical']

# Print support values for books, books_imported, and books_technical.
onehot[['books','books_imported','books_technical']].mean(axis=0)


# ### **Compute the confidence metric**

# * **The support metric doesn't provide information about direction.**
#   * $support(antecedent, consequent) = support(consequent, antecedent)$
# 
# * **The confidence metric has a direction.**
#   * Conditional probability of the consequent, given the antecedent.

# $$confidence(antecedent \rightarrow consequent)= \frac{support(antecedent, consequent)}{support(antecedent)}$$

# * A high value of confidence indicates that the antecedent and consequent are associated and that the direction of the association runs from the antecedent to the consequent.

# In[32]:


# Compute joint support for sports_leisure and health_beauty.
joint_support = (onehot['sports_leisure'] & onehot['health_beauty']).mean()

# Print confidence metric for sports_leisure -> health_beauty.
joint_support / onehot['sports_leisure'].mean()


# In[33]:


# Print confidence for health_beauty -> sports_leisure.
joint_support / onehot['sports_leisure'].mean()


# **Insight:** $confidence(sports\_leisure \rightarrow health\_beauty)$ was higher than $confidence(health\_beauty \rightarrow sports\_leisure)$. Since the two have the same joint support, the confidence measures will differ only by the antecedent support. The higher confidence metric means that the antecedent has *lower* support.

# ---
# <center><h1> Q&A 3</h1> </center>
# 
# ---

# ## **The Apriori Algorithm and Pruning**

# **The Apriori algorithm** identifies frequent (high support) itemsets using something called the Apriori principle, which states that a superset that contains an infrequent item is also infrequent.

# ![alt](https://github.com/datacamp/Market-Basket-Analysis-in-python-live-training/blob/master/assets/apriori_algorithm.png?raw=True)

# **Pruning** is the process of removing itemsets or association rules, typically based on the application of a metric threshold. 

# **The `mlxtend` module will enable us to apply the Apriori algorithm, perform pruning, and compute association rules.**

# ### **Applying the Apriori algorithm**

# * Use `apriori()` to identify frequent itemsets.
# * `min_support` set the item frequency threshold used for pruning.

# In[34]:


from mlxtend.frequent_patterns import apriori

# Apply apriori algorithm to data with min support threshold of 0.01.
frequent_itemsets = apriori(onehot, min_support = 0.01)

# Print frequent itemsets.
frequent_itemsets


# **Observation 1:** `apriori` returns a `DataFrame` with a `support` column and an `itemsets` column.
# 
# **Observation 2:** By default `apriori` returns itemset numbers, rather than labels. We can change this by using the `use_colnames` parameter.
# 
# **Insight:** All itemsets with a support of greater than 0.01 contain a single item.

# * Use `use_colnames` to use item names, rather than integer IDs.

# In[35]:


# Apply apriori algorithm to data with min support threshold of 0.001.
frequent_itemsets = apriori(onehot, min_support = 0.001, use_colnames = True)

# Print frequent itemsets.
frequent_itemsets


# **Insight:** Lowering the support threshold increased the number of itemsets returned and even yielded itemsets with more than one item.

# In[36]:


# Apply apriori algorithm to data with min support threshold of 0.00005.
frequent_itemsets = apriori(onehot, min_support = 0.00005, use_colnames = True)

# Print frequent itemsets.
frequent_itemsets


# **Observation:** Notice how low we must set the support threshold (0.005%) to return a high number of itemsets with more than one item.

# In[37]:


# Apply apriori algorithm to data with a two-item limit.
frequent_itemsets = apriori(onehot, min_support = 0.00005, max_len = 2, use_colnames = True)


# **Insight:** What do we gain from the apriori algorithm? We start off with $2^{71}$ potential itemsets and immediately reduce it to 113 without enumerating all $2^{71}$ itemsets.

# ### **Computing association rules from Apriori output**

# * Use `association_rules()` to compute and prune association rules from output of `apriori()`.

# In[38]:


from mlxtend.frequent_patterns import association_rules

# Recover association rules using support and a minimum threshold of 0.0001.
rules = association_rules(frequent_itemsets, metric = 'support', min_threshold = 0.0001)

# Print rules header.
rules.head()


# **Notice that `association_rules` automatically computes seven metrics.**

# ### **Pruning association rules**

# In[39]:


# Recover association rules using confidence threshold of 0.01.
rules = association_rules(frequent_itemsets, metric = 'confidence', min_threshold = 0.01)

# Print rules.
rules


# In[40]:


# Select rules with a consequent support above 0.095.
rules = rules[rules['consequent support'] > 0.095]

# Print rules.
rules


# ### **The leverage metric**
# 
# * **Leverage provides a sanity check.**
#   * $support(antecedent, consequent)$ = joint support in data.
#   * $support(antecedent) * support(consequent)$ = expected joint support for unrelated antecedent and consequent.

# * **Leverage formula**
#   * $$leverage(antecendent, consequent) = 
# support(antecedent, consequent) - support(antecedent) * support(consequent)$$

# * **For most problems, we will discard itemsets with negative leverage.**
#   * Negative leverage means that the items appear together less frequently than we would expect if they were randomly and independently distributed across transactions.

# In[41]:


# Select rules with leverage higher than 0.0.
rules = rules[rules['leverage'] > 0.0]

# Print rules.
rules


# **Insight:** The Apriori algorithm reduced the number of itemsets from $2^{71}$ to 113. Pruning allowed us to identify to a single association rule that could be useful for cross-promotional purposes: $\{home\_comfort\} \rightarrow \{bed\_bath\_table\}$.

# ### **Visualizing patterns in metrics**

# * `sns.scatterplot()` creates a scatterplot from two columns in a `DataFrame`.

# In[42]:


# Recover association rules with a minimum support greater than 0.000001.
rules = association_rules(frequent_itemsets, metric = 'support', min_threshold = 0.000001)

# Plot leverage against confidence.
plt.figure(figsize=(15,5))
sns.scatterplot(x="leverage", y="confidence", data=rules)


# **Insight 1**: Leverage and confidence contain some of the same information about the strength of an association.

# ---
# <center><h1> Q&A 4</h1> </center>
# 
# ---

# In[43]:


# Experiment on your own here.

