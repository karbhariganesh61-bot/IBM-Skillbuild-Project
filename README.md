# Classic Models Dataset — Analytics Project

A full end-to-end analytics project on the Classic Models toy-car company dataset, covering data cleaning and an interactive browser dashboard.

---

## Project Structure

```
classic_models_dataset/       ← raw CSV source files
  customers.csv
  employees.csv
  offices.csv
  order details.csv
  orders.csv
  payments.csv
  productlines.csv
  products.csv

cleaned_data/                 ← output of clean_data.py (generated)
  customers.csv
  employees.csv
  offices.csv
  orders.csv
  order_details.csv
  payments.csv
  productlines.csv
  products.csv

clean_data.py                 ← Python data cleaning script
dashboard.html                ← Interactive analytics dashboard
README.md                     ← This file
```

---

## Dataset Overview

| Table | Rows | Key Columns |
|---|---|---|
| customers | 122 | customerNumber, customerName, country, creditLimit |
| employees | 23 | employeeNumber, jobTitle, officeCode, reportsTo |
| offices | 7 | officeCode, city, country, territory |
| orders | 326 | orderNumber, orderDate, status, customerNumber |
| order details | 2 996 | orderNumber, productCode, quantityOrdered, priceEach |
| payments | 273 | customerNumber, checkNumber, paymentDate, amount |
| productlines | 7 | productLine, textDescription |
| products | 110 | productCode, productName, productLine, buyPrice, MSRP |

---

## Data Cleaning (`clean_data.py`)

### Requirements
```
pip install pandas
```

### Run
```bash
python clean_data.py
```

Cleaned CSVs are written to `./cleaned_data/`.

### Cleaning Rules Applied

| Table | Issues Found | Fix Applied | Computed Columns Added |
|---|---|---|---|
| customers | 202 literal `NULL` strings in state, addressLine2, salesRepEmployeeNumber | Replace with empty string / NaN; cast to correct types | — |
| employees | 1 literal `NULL` in reportsTo | Replace with NaN; cast to Int64 | — |
| offices | 5 literal `NULL` in addressLine2, state | Replace with empty string | — |
| orders | 260 literal `NULL` in shippedDate (not-yet-shipped) and comments | shippedDate → NaT; comments → ""; dates cast to datetime | — |
| order details | 0 nulls | Cast quantityOrdered, priceEach to numeric | **lineTotal** = qty × price |
| payments | 0 nulls | Cast paymentDate to datetime, amount to float | — |
| productlines | 14 literal `NULL` in htmlDescription, image | Replace with empty string | — |
| products | 0 nulls | Cast quantityInStock, buyPrice, MSRP to numeric | **margin** = MSRP − buyPrice; **marginPct** |

---

## Interactive Dashboard (`dashboard.html`)

Open directly in any modern browser — no server required.

```bash
# Windows
start dashboard.html

# macOS
open dashboard.html

# Linux
xdg-open dashboard.html
```

### Dashboard Tabs

| Tab | What You'll See |
|---|---|
| **Overview** | KPI cards, monthly revenue trend, product-line pie, order status donut, payments by year, customers by country |
| **Revenue** | Year filter, filtered monthly bar chart, cumulative revenue line, revenue vs units by product line |
| **Products** | Product-line filter, top-10 products bar, units-sold pie, full leaderboard table with inline bars |
| **Customers** | Top-10 customer bar, country distribution pie, payment leaderboard table |
| **Orders** | Status KPIs, status donut, monthly order volume bar |
| **Data Quality** | NULL-string audit bar chart, cleaning rules reference table |

### Key Metrics

| Metric | Value |
|---|---|
| Total order-line revenue | **$9.60 M** |
| Total payments received | **$8.85 M** |
| Total orders | **326** (93% Shipped) |
| Date range | Jan 2003 – May 2005 |
| Top product line | Classic Cars ($3.85 M) |
| Top customer | Euro+ Shopping Channel ($715 K) |

---

## Tech Stack

- **Python 3 + pandas** — data cleaning
- **HTML5 + CSS3 + ECharts 5** — interactive dashboard (zero dependencies beyond the CDN)
