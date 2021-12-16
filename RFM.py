import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.5f' % x)
################################################################
# data analysing and data process
################################################################

df_ = pd.read_excel("dataset/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.describe().T

df.isnull().sum()

df.dropna(inplace = True)

df["StockCode"].nunique()

df["StockCode"].value_counts()

df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending=False)
df[~df["Invoice"].str.contains("C", na = False)]

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

################################################################
# RFM calculation of metrics
################################################################


today_date = dt.datetime(2011,12,11)
rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date-InvoiceDate.max()).days,
                                    "Invoice": lambda Invoice: Invoice.nunique(),
                                    "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ["Recency","Frequency","Monetary"]
rfm = rfm.loc[(rfm["Monetary"] > 0)]

################################################################
# RFM make scores ve converting to a single variable
################################################################



rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels = (5,4,3,2,1))
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method = "first"), 5, labels = (1,2,3,4,5))
rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=(5,4,3,2,1))
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str))

################################################################
# RFM definition of scores as segments
################################################################


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)



################################################################
# Action time
################################################################


rfm.loc[(rfm["segment"] == "at_Risk") | (rfm["segment"] == "need_attention") | (rfm["segment"] == "hibernating")].groupby("segment").agg({"mean","std"})
new_df = pd.DataFrame()
new_df["loyal_customers"] = rfm[rfm["segment"] == "loyal_customers"].index
new_df.to_excel("loyalCustomer.xlsx")
rfm.head()

############################################
# CUSTOMER LIFETIME VALUE
############################################

# 1. Veri Hazırlama
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Segmentlerin Oluşturulması

##################################################
# 1. Data Processing
##################################################

cltv_c = df.groupby(["Customer ID"]).agg({"Invoice": lambda x: x.nunique(),
                                 "Quantity": lambda x: x.sum(),
                                 "TotalPrice": lambda x: x.sum()})

cltv_c.columns = ["total_transaction","total_unit","total_price"]
cltv_c.head()

##################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################

cltv_c["avarage_order_value"] = cltv_c["total_price"]/cltv_c["total_transaction"]

##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

##################################################
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
##################################################

repeat_rate = (cltv_c[cltv_c.total_transaction>1].shape[0] / cltv_c.shape[0])
churn_rate = 1-repeat_rate

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

####################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# ##################################################

cltv_c["customer_value"] = cltv_c["avarage_order_value"] * cltv_c["purchase_frequency"]

# ##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# ##################################################

cltv_c["CLTV"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_c[["CLTV"]])
cltv_c["scaled_cltv"] = scaler.transform(cltv_c[["CLTV"]])

# ##################################################
# 8.Creating Segments
# ##################################################

cltv_c["segment"] = pd.qcut(cltv_c["scaled_cltv"], 4, labels=("D", "C","B","A"))
cltv_c.head()