
## Project Title
**Market Basket Analysis and Product Recommendation System using Apriori Algorithm**

---

# Project Overview

This project performs **Market Basket Analysis** using the **Apriori Algorithm** to identify hidden relationships between products frequently purchased together in retail transactions.

The analysis helps businesses:
- Discover customer purchasing patterns
- Improve cross-selling strategies
- Create effective product bundles
- Optimize store layouts
- Generate personalized product recommendations
- Improve customer experience and sales performance

The project was implemented using **Python**, **Google Colab**, and association rule mining techniques.

---

# Business Problem

Retail businesses generate huge volumes of transactional purchase data every day. However, manually identifying relationships between products is difficult.

The business wants to understand:

- Which products are frequently purchased together
- Which products should be bundled together
- Which products improve cross-selling opportunities
- Which products should be placed near each other
- Which products should be recommended to customers

The objective of this project is to apply **Market Basket Analysis** techniques to identify meaningful product associations and generate business recommendations.

---

# Dataset Description

The dataset used in this project is a retail transaction dataset containing customer purchase records.

## Dataset Columns

| Column Name | Description |
|---|---|
| TransactionID | Unique identifier for each transaction |
| CustomerID | Unique customer identifier |
| TransactionDate | Date of transaction |
| ProductID | Unique identifier for each product |
| ProductName | Name of purchased product |
| Quantity | Quantity purchased |
| UnitPrice | Price per unit |

---

# Data Understanding

## What is a Transaction?

A transaction represents a single customer purchase event identified using a transaction ID.

### Example

| Transaction ID | Purchased Products |
|---|---|
| T001 | Bread, Milk, Butter |

---

## What is an Item?

An item refers to an individual product purchased within a transaction.

### Examples
- Bread
- Milk
- Butter

---

## What Each Row Represents

Each row in the dataset represents one purchased product within a transaction.

If multiple products are purchased in a single transaction, multiple rows share the same transaction ID.

---

# Why Market Basket Analysis is Important

Market Basket Analysis helps businesses identify products that customers frequently purchase together.

These insights help businesses:
- Improve product recommendations
- Increase sales
- Optimize store layouts
- Design targeted promotions
- Enhance customer experience

---

# Cross-Selling and Upselling

## Cross-Selling

Cross-selling means recommending related products based on current purchases.

### Example

If customers frequently buy:

**Dishwash Liquid → Fabric Softener**

the business can recommend Fabric Softener whenever a customer purchases Dishwash Liquid.

---

## Upselling

Upselling means encouraging customers to purchase premium or upgraded products.

### Example

Customers purchasing a coffee machine may also be encouraged to buy premium coffee capsules.

---

# Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Mlxtend
- Matplotlib
- Seaborn

---

# Data Cleaning Summary

Several preprocessing operations were performed before generating association rules.

## Cleaning Steps Performed

### 1. Removed Missing Product Names

```python
df = df.dropna(subset=['ProductName'])
````

### 2. Removed Duplicate Records

```python
df = df.drop_duplicates()
```

### 3. Removed Cancelled Transactions

```python
df = df[~df['TransactionID'].astype(str).str.startswith('C')]
```

### 4. Removed Invalid Quantities

```python
df = df[df['Quantity'] > 0]
```

### 5. Checked Rare Products

Products purchased very few times were analyzed.

```python
df['ProductName'].value_counts()
```

### 6. Corrected Datatypes

```python
df['TransactionID'] = df['TransactionID'].astype(str)
df['Quantity'] = df['Quantity'].astype(int)
```

---

# Basket Preparation Method

The dataset was converted into basket format suitable for Market Basket Analysis.

## Basket Transformation Process

The following steps were performed:

* Grouped data using:

  * TransactionID
  * ProductName

* Converted transactional data into matrix format

* Applied binary encoding:

  * 1 = Product Purchased
  * 0 = Product Not Purchased

---

# Basket Creation Code

```python
basket = (
    df.groupby(['TransactionID', 'ProductName'])['Quantity']
    .sum()
    .unstack()
    .fillna(0)
)

basket = basket.applymap(lambda x: 1 if x > 0 else 0)
```

---

# Basket Format Example

| Transaction ID | Bread | Milk | Butter |
| -------------- | ----- | ---- | ------ |
| T001           | 1     | 0    | 1      |
| T002           | 0     | 1    | 1      |

---

# Frequent Itemset Summary

Frequent itemsets were generated using the **Apriori Algorithm**.

## Support Values Tested

| Support Value | Observation                           |
| ------------- | ------------------------------------- |
| 0.01          | Generated many weak itemsets          |
| 0.05          | Balanced and meaningful itemsets      |
| 0.10          | Generated fewer but stronger itemsets |

A support value of **0.05** was selected because it provided meaningful product combinations without excessive noise.

---

# Frequent Itemset Generation Code

```python
min_support_chosen = 0.05

frequent_itemsets = apriori(
    basket,
    min_support=min_support_chosen,
    use_colnames=True
)

display(
    frequent_itemsets
    .sort_values(by='support', ascending=False)
    .head(10)
)
```

---

# Association Rules Summary

Association rules were generated using:

* Support
* Confidence
* Lift

---

# Association Rule Generation Code

```python
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

display(
    rules[
        ['antecedents', 'consequents',
         'support', 'confidence', 'lift']
    ]
)
```

---

# Understanding Association Metrics

## Support

Support measures how frequently an itemset appears in the dataset.

### Example

If support for:

{Milk, Bread} = 0.10

then 10% of all transactions contain both Milk and Bread.

---

## Confidence

Confidence measures how often the consequent appears when the antecedent is present.

### Example

If confidence for:

{Milk} → {Bread} = 0.80

then 80% of customers purchasing Milk also purchase Bread.

---

## Lift

Lift measures how strongly two products are associated beyond random chance.

### Interpretation

| Lift Value | Meaning              |
| ---------- | -------------------- |
| Lift = 1   | No association       |
| Lift > 1   | Positive association |
| Lift < 1   | Negative association |

Higher lift values indicate stronger product relationships.

---

# Top Association Rules with Interpretation

## 1. Dishwash Liquid → Fabric Softener

* Confidence: 0.702
* Lift: 8.366

Customers purchasing Dishwash Liquid are highly likely to purchase Fabric Softener as well.

---

## 2. Nachos → Salsa Dip

* Confidence: 0.770
* Lift: 7.965

Customers buying Nachos frequently purchase Salsa Dip together.

---

## 3. Potato Chips → Salsa Dip

* Confidence: 0.731
* Lift: 7.562

Potato Chips and Salsa Dip form a strong snack combination.

---

## 4. Butter + Bread Loaf → Chocolate Spread

* Confidence: 0.868
* Lift: 7.229

Customers purchasing Butter and Bread Loaf are very likely to buy Chocolate Spread.

---

# Final Business Recommendations

Based on the generated association rules, the following business recommendations were identified.

---

# Product Bundling

## 1. Dishwash Liquid & Fabric Softener

Create:

* Household Essentials Bundle
* Cleaning Combo Offers

---

## 2. Nachos & Salsa Dip

Create:

* Snack Night Bundle
* Movie Night Combo

---

## 3. Potato Chips & Salsa Dip

Create:

* Party Snack Pack
* Weekend Combo Offer

---

## 4. Bread Loaf, Butter & Chocolate Spread

Create:

* Breakfast Trio Bundle
* Morning Essentials Combo

---

# Product Placement

## Recommended Placements

| Products                          | Placement Strategy       |
| --------------------------------- | ------------------------ |
| Dishwash Liquid & Fabric Softener | Same cleaning aisle      |
| Nachos, Chips & Salsa Dip         | Same snack section       |
| Bread, Butter & Chocolate Spread  | Breakfast corner display |

---

# Cross-Selling Strategies

* Recommend Fabric Softener when customers buy Dishwash Liquid
* Recommend Salsa Dip with Nachos or Potato Chips
* Recommend Butter and Chocolate Spread with Bread Loaf

---

# Promotional Campaigns

## Household Cleaning Campaign

* Dishwash Liquid + Fabric Softener discounts

## Snack Combo Campaign

* Nachos/Chips + Salsa Dip offers

## Breakfast Essentials Campaign

* Bread + Butter + Chocolate Spread promotions

---

# Improving Customer Experience

These insights help businesses:

* Increase basket size
* Improve product recommendations
* Enhance store layout
* Optimize inventory management
* Improve targeted marketing

---

# Visualizations Included

The project includes visualizations such as:

* Frequent Itemsets vs Support Threshold
* Association Rules Scatter Plot
* Support vs Confidence Graph
* Top Association Rules Table

---

# Project Structure

```bash
Market-Basket-Analysis/
│
├── Market_Basket_Analysis.ipynb
├── README.md
├── requirements.txt
├── dataset_source.md
├── outputs/
└── images/
```

---

# How to Run the Project

## Step 1: Clone Repository

```bash
git clone <repository-link>
```

---

## Step 2: Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Step 3: Open Jupyter Notebook or Google Colab

Open:

```bash
Market_Basket_Analysis.ipynb
```

---

## Step 4: Run All Cells

Execute all notebook cells sequentially.

---

# Google Colab Notebook

You can also view and run the project directly in Google Colab using the notebook link included in this repository.

---

# Conclusion

This project successfully applied the Apriori Algorithm to identify meaningful product associations from retail transaction data.

The generated insights can help businesses:

* Improve cross-selling
* Design effective product bundles
* Optimize store layout
* Increase sales
* Enhance customer satisfaction

The project demonstrates how Market Basket Analysis can support data-driven business decisions in the retail industry.

```
```
