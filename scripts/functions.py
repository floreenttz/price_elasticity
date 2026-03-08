import pandas as pd
import numpy as np
import io
from pandas.tseries.offsets import MonthBegin
import boto3


def round_2_dec(n):
    return round(n, 2)

def calculate_median(series):
    median_value = np.median(series)
    return pd.Series({'Median': median_value})


# Define function to generate sequence
def generate_sequence(row):
    return list(range(row["start_indices"], row["end_indices"] + 1))


# Add the condition for calculating 'min_price_level'
def calculate_min_price_level(row):
    if row["price_80_perc"] < row["price_below_10_perc"]:
        return (
            round((row["price_80_perc"] / row["price_below_10_perc"]) * 100 - 100) - 10
        ) / 100
    else:
        return -10 / 100


def calc_prices(last_price, min_price_level):
    price_percents = np.array(list(np.arange(min_price_level, 0.11, 0.01)))
    return list(map(round_2_dec, last_price + (last_price * price_percents)))


def detect_separator(file_path, sample_size=10):
    """
    Detects the CSV separator used in the file by analyzing a sample of the file.

    Parameters:
        file_path (str): Path to the CSV file.
        sample_size (int): Number of rows to use for separator detection (default: 10).

    Returns:
        str: Detected CSV separator (e.g., ',', ';', '\t').
    """
    # Read a sample of the file to detect the separator
    s3 = boto3.client('s3')
    bucket_name = file_path.split('/')[2]
    key = "/".join(file_path.split('/')[3:])

    # Download the sample data from S3
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    sample_data = obj['Body'].read().decode('utf-8').splitlines()[:sample_size]
    sample_data = "\n".join(sample_data)

    # Try detecting potential separators
    potential_separators = [",", ";", "\t"]  # List of potential separators to test

    for sep in potential_separators:
        # Count the number of columns for each potential separator
        try:
            df = pd.read_csv(io.StringIO(sample_data), sep=sep)
            if len(df.columns) > 1:
                return sep  # Return the detected separator if more than one column is found
        except pd.errors.ParserError:
            continue  # Continue to the next potential separator if parsing fails

    # Default to ',' if no separator is detected
    return ","  # Default separator (comma) if detection fails


def rename_columns(data):

    column_mapping = {
        "nr_of_stores_per_month": "number_of_stores",
        "promotion_ind": "promotion_indicator",
        "promo_ind": "promotion_indicator",
        "product_category_code_level1": "category_code_level1",
        "product_category_code_level2": "category_code_level2",
        "product_category_code_level3": "category_code_level3",
        "product_category_name_level1": "category_name_level1",
        "product_category_name_level2": "category_name_level2",
        "product_category_name_level3": "category_name_level3",
        "product_category_level1": "category_name_level1",
        "product_category_level2": "category_name_level2",
        "product_category_level3": "category_name_level3",
        "revenue_after_discount_incl_vat": "revenue_after",
        "revenue_before_discount_incl_vat": "revenue_before",
        "albert_heijn_price": "ah_price",
        "total_quantity_sold": "quantity_sold",
        "total_revenue_before_discount": "revenue_before",
        "total_revenue_after_discount": "revenue_after",
        "number_of_stores_per_week": "number_of_stores",
        "category_level_4": "category_name_level4",
        "category_level_3": "category_name_level3",
        "category_level_2": "category_name_level2", 
        "category_level_1": "category_name_level1",
        # Add more mappings for other column aliases as needed
    }

    data = data.rename(columns=column_mapping)
    return data


def cpi_regex(cpi, columns):

    cpi.month_year = cpi.month_year.astype(str).str.replace("\*", "", regex=True)
    cpi = cpi[~cpi["month_year"].str.match(r"[0-9]+$")].reset_index(drop=True)
    cpi = cpi.fillna(method="ffill")
    cpi.columns = columns
    cpi.cpi = cpi.cpi.str.replace(",", ".", regex=True).astype(float)

    month_mapping = {
        "januari": "January",
        "februari": "February",
        "maart": "March",
        "april": "April",
        "mei": "May",
        "juni": "June",
        "juli": "July",
        "augustus": "August",
        "september": "September",
        "oktober": "October",
        "november": "November",
        "december": "December",
    }

    # Replace Dutch month names with English month names
    for dutch, english in month_mapping.items():

        cpi["month_year"] = cpi["month_year"].str.replace(dutch, english, regex=True)
    cpi.month_year = pd.to_datetime(cpi.month_year, format="%Y %B", errors="coerce")

    return cpi.dropna()


def cpi_date_columns(cpi):
    cpi["month_year"] = pd.to_datetime(cpi["month_year"])
    cpi["year"] = cpi["month_year"].dt.year
    cpi["month"] = cpi["month_year"].dt.month
    cpi["days_in_month"] = cpi["month_year"].dt.days_in_month
    return cpi


def actualize_cpi(df, has_categories=True):

    current_time = pd.Timestamp.now().normalize()
    next_month = current_time + MonthBegin(1)
    max_date_in_df = df["month_year"].max()
    new_dates = pd.date_range(
        start=max_date_in_df + MonthBegin(1), end=next_month, freq="MS"
    )

    if has_categories:
        categories = df["categories"].unique()
        last_known_cpi_per_category = (
            df.dropna(subset=["cpi"]).groupby("categories")["cpi"].last().to_dict()
        )

        new_rows = [
            {
                "categories": cat,
                "month_year": date,
                "cpi": last_known_cpi_per_category.get(cat),
            }
            for cat in categories
            for date in new_dates
        ]
    else:
        last_known_cpi = df.dropna(subset=["cpi"])["cpi"].iloc[-1]
        new_rows = [{"month_year": date, "cpi": last_known_cpi} for date in new_dates]

    new_df = pd.DataFrame(new_rows)
    return pd.concat([df, new_df], ignore_index=True)


def interpolate_cpi(data):
    daily_data = []
    has_category = "category_name" in data.columns

    # Ensure month_year is a datetime object
    data['month_year'] = pd.to_datetime(data['month_year'])

    for i in range(len(data) - 1):
        # Generate daily dates between two months, excluding the last day of the second month
        daily_dates = pd.date_range(
            start=data['month_year'].iloc[i], end=data['month_year'].iloc[i + 1], freq="D"
        )[:-1]
        interpolated_values = np.linspace(
            data['cpi'].iloc[i], data['cpi'].iloc[i + 1], len(daily_dates)
        )

        if has_category:
            daily_data.append(
                pd.DataFrame(
                    {
                        "category_name": data['category_name'].iloc[i],
                        "date": daily_dates,
                        "cpi_per_category": interpolated_values,
                    }
                )
            )
        else:
            daily_data.append(
                pd.DataFrame({"date": daily_dates, "cpi": interpolated_values})
            )

    # Handle the last month's value properly by filling it to the end of the month
    last_month_start = data['month_year'].iloc[-1]
    last_month_end = last_month_start + pd.offsets.MonthEnd(0)
    last_dates = pd.date_range(start=last_month_start, end=last_month_end, freq='D')
    last_values = np.full(len(last_dates), data['cpi'].iloc[-1])

    if has_category:
        daily_data.append(
            pd.DataFrame(
                {
                    "category_name": data['category_name'].iloc[-1],
                    "date": last_dates,
                    "cpi_per_category": last_values,
                }
            )
        )
    else:
        daily_data.append(
            pd.DataFrame({"date": last_dates, "cpi": last_values})
        )

    return pd.concat(daily_data).reset_index(drop=True)


# def interpolate_cpi(group):
    
#     daily_data = []
#     has_category = "categories" in group.columns

#     for i in range(len(group) - 1):
#         daily_dates = pd.date_range(
#             start=group.month_year.iloc[i], end=group.month_year.iloc[i + 1], freq="D"
#         )[:-1]
#         interpolated_values = np.linspace(
#             group.cpi.iloc[i], group.cpi.iloc[i + 1], len(daily_dates)
#         )

#         if has_category:
#             daily_data.append(
#                 pd.DataFrame(
#                     {
#                         "category": group.categories.iloc[i],
#                         "date": daily_dates,
#                         "cpi_per_category": interpolated_values,
#                     }
#                 )
#             )
#         else:
#             daily_data.append(
#                 pd.DataFrame({"date": daily_dates, "cpi": interpolated_values})
#             )
    
#       # Handle the last month's value properly
#     if len(group) > 1:
#         last_month_start = group.month_year.iloc[-1]
#         last_month_end = (last_month_start + pd.offsets.MonthEnd(0)).to_timestamp()
#         last_dates = pd.date_range(start=last_month_start, end=last_month_end, freq='D')
#         last_values = np.full(len(last_dates), group.cpi.iloc[-1])

#         if has_category:
#             daily_data.append(
#                 pd.DataFrame(
#                     {
#                         "category": group.categories.iloc[-1],
#                         "date": last_dates,
#                         "cpi_per_category": last_values,
#                     }
#                 )
#             )
#         else:
#             daily_data.append(
#                 pd.DataFrame({"date": last_dates, "cpi": last_values})
#             )

#     return pd.concat(daily_data)




def elasticity_mapping(data, column):
    
    # Define a mapping from subcategories to broader categories
    category_mapping = {
        
        # Baby & Kids
        'Baby-, kindervoeding': 'Baby & Kids',

        # Baking & Desserts
        'Bakproducten en desser': 'Baking & Desserts',
        # Bread Alternatives
        'Broodvervangers': 'Baking & Desserts',


        # Beverages
        'Bieren': 'Biers',
        
        'Niet alcoholhoudende d': 'Non-alcoholic drinks',
        
        'Wijnen en aperitieven': 'Wine and aperitifs',
        
        'Sterk alcoholische dra': 'Alcoholic drinks',
        'Dagverse drinks': 'Alcoholic drinks',

        
        'Koffie, cacao': 'Coffee, cacao and tea',
        'Thee': 'Coffee, cacao and tea',

        # Spreads & Butters
        'Boterhambeleg': 'Spreads & Butters',

        
        # International Cuisine
        'Buitenlandse specialit': 'Cuisine',
        'Internationale straat': 'Cuisine',
        'Oosters': 'Cuisine',

        # Snacks & Sweets
        'Chips': 'Snacks & Sweets',
        'Vleessnacks': 'Snacks & Sweets',
        'Chocolade': 'Snacks & Sweets',
        'Suikerwerk': 'Snacks & Sweets',
        'Koek & biscuit': 'Snacks & Sweets',
        
        'IJs': 'Ice cream',

        # Personal Care
        'Damesverzorging': 'Personal Care',
        'Haarverzorging': 'Personal Care',
        'Mondverzorging': 'Personal Care',
        # Health & Wellness
        'Bewust en verantwoord': 'Personal Care',


        # Frozen Meals
        'Diepvries maaltijden': 'Frozen Meals',

        # Drugstore Items
        'Drogisterij': 'Drugstore Items',

        # Dairy
        'Eetzuivel naturel': 'Dairy',
        'Eetzuivel toevoeging': 'Dairy',
        'Houdbare drinkzuivel': 'Dairy',

        # Sauces & Condiments
        'Gele, Rode, Warme Sauzen': 'Sauces & Condiments',
        'Maaltijdsauzen': 'Sauces & Condiments',
        'Kruiden': 'Sauces & Condiments',

        # Canned & Preserved Foods
        'Groenteconserven': 'Canned & Preserved Foods',
        'Soepen en bouillons': 'Canned & Preserved Foods',

        # Pet Food
        'Hondenvoer': 'Pet Food',
        'Kattenvoer': 'Pet Food',

        # Cleaners
        'Reinigers': 'Household Items',
        'Wasverzorging': 'Household Items',

        # Rice & Pasta
        'Rijst en deegwaren': 'Rice & Pasta',

        # Fresh Products
        'Snijverse vleeswaren': 'Meat Products',
        'Vers vlees': 'Meat Products',
        
        'Vers dinerproducten': 'Fresh Products',
        'Vers fruit': 'Fresh Products',
        'Verse groenten': 'Fresh Products',
        
        'Verse bloemen en plant': 'Plants',

        # Value Packs
        'Voordeelpakkers': 'Value Packs',

        # Household Miscellaneous
        'huishoud overig': 'Household Miscellaneous',

        # Tobacco Products
        'Tabaksproducten': 'Tobacco Products'
    }

    # Map the subcategories to broader categories
    data[column] = data[column].map(category_mapping)
    return(data)

def add_sequence_numbers_to_last_days(last_days_df, num_sequences):
    last_days_df['sequence_number'] = last_days_df.groupby('product_code').cumcount() + 1
    last_days_df['sequence_number'] = last_days_df['sequence_number'] % num_sequences
    last_days_df['sequence_number'].replace(0, num_sequences, inplace=True)
    return last_days_df


def remove_outliers(group, column):
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = max(Q1 - 2 * IQR, 0)
    upper_bound = Q3 + 2 * IQR

    # Filter the group based on the bounds
    return group[(group[column] >= lower_bound) & 
                 (group[column] <= upper_bound)]



def fill_max_2_consecutive_nans(df, product_column, price_column, max_fill_length=2):
    # Function to fill up to max_fill_length consecutive NaNs with the previous value
    def fill_nans(group):
        filled_values = group[price_column].copy()
        nans = filled_values.isna()
        start = None
        
        for i, is_nan in enumerate(nans):
            if is_nan and start is None:
                start = i  # Start of NaN sequence
            elif not is_nan and start is not None:
                end = i  # End of NaN sequence
                if end - start <= max_fill_length:
                    # Fill NaNs if the length is within the limit
                    if start > 0:
                        filled_values.iloc[start:end] = filled_values.iloc[start - 1]
                start = None
        
        # Check if the last values were NaN and within the limit
        if start is not None:
            end = len(filled_values)
            if end - start <= max_fill_length:
                if start > 0:
                    filled_values.iloc[start:end] = filled_values.iloc[start - 1]
        
        return filled_values

    # Apply the fill_nans function to each group and return the updated DataFrame
    df[price_column] = df.groupby(product_column).apply(fill_nans).reset_index(level=0, drop=True)
    return df[price_column]


def calculate_sales_fraction(df, column):
    # Calculate total sales per category
    total_sales_category = df['quantity_sold'].sum()
    
    # Calculate total sales per product in the category
    total_sales_per_product = df.groupby('product_code')['quantity_sold'].sum()
    
    # Compute the fraction of sales for each product within the category
    sales_fraction = total_sales_per_product / total_sales_category
    
    # Merge the sales fraction back to the DataFrame
    result = df.drop_duplicates(subset=['product_code']).copy()
    result['total_sales'] = result['product_code'].map(total_sales_per_product)
    result['sales_fraction'] = result['product_code'].map(sales_fraction)
    
    return result[[column, 'product_code', 'total_sales', 'sales_fraction']]

def compute_price_per_kg(data):

    # Calculate price_per_kg or fill with product_selling_price
    data['price_per_kg'] = np.where(data['weight_sold'] > 0, 
                                    np.round(data['calculated_price']* data['quantity_sold'] / data['weight_sold'], 2), 
                                    data['product_selling_price'])
    tolerance = 0.5
    data['is_equal_price'] = (abs(data['price_per_kg'] - data['product_selling_price']) < tolerance).astype(int)
    return(data)

# Correct aggregation: check if a price change occurs on the first day of the week
def custom_aggregation(group):
    # Check if the group is not empty
    if len(group) == 0:
        return None
    # If a price change occurs on the first day of the week, reset to 0
    if group['price_change'].iloc[0] == 1:
        return 0
    else:
        return group['days_since_last_change'].iloc[-1]
    

def fill_except_median(group, column_name):
    # Find the median index for the current group
    median_index = group.index[len(group) // 2]
    
    # Set all values in the specified column to 0, except the median row
    group.loc[group.index != median_index, column_name] = 0  # Set all rows except the median to 0
    
    return group
