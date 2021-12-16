import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import mysql.connector



pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option("display.max_columns", None)



df_ = pd.read_excel("dataset/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

########################################
# Verinin hazırlanması
########################################

df.dropna(inplace = True)
df = df[~df["Invoice"].str.contains("C", na = False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df["total_price"] = df["Quantity"] * df["Price"]
df = df[df["Country"] == "United Kingdom"]



today_date = dt.datetime(2011,12,11)
cltv = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                      lambda date: (today_date - date.min()).days],
                                      'Invoice': lambda num: num.nunique(),
                                      'total_price': lambda TotalPrice: TotalPrice.sum()})



cltv.columns = cltv.columns.droplevel(0)
cltv.columns = ["recency","T","frequency","monetary"]
cltv = cltv[cltv["monetary"] > 0]
cltv = cltv[(cltv['frequency'] > 1)]
cltv["recency"] = cltv["recency"]/7
cltv["T"] = cltv["T"]/7
cltv["monetary"] = cltv["monetary"]/cltv["frequency"]

############################################
# Veritabanına Bağlanma
############################################

creds = {'user': 'group_03',
         'passwd': 'hayatguzelkodlarucuyor',
         'host': '34.88.156.118',
         'port': 3306,
         "db":"group_03"}


connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'


conn = create_engine(connstr.format(**creds))



############################################
# BGNBD Modelinin Kurulması
############################################

bgf = BetaGeoFitter(penalizer_coef = 0.001)
bgf.fit(cltv["frequency"],
        cltv["recency"],
        cltv["T"])

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv['frequency'],
                                                        cltv['recency'],
                                                        cltv['T']).sort_values(ascending=False).head(10)
cltv["expected_purc_1_week"] = bgf.predict(1,
                                              cltv['frequency'],
                                              cltv['recency'],
                                              cltv['T'])

cltv["expected_purc_1_month"] = bgf.predict(4,
                                              cltv['frequency'],
                                              cltv['recency'],
                                              cltv['T'])

cltv.sort_values("expected_purc_1_month", ascending=False).head(20)

#################################################
# GammaGamma Modelinin Kurulması
#################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv['frequency'], cltv['monetary'])

cltv["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                          cltv['monetary'])

#####################################################
# BGNBD && GammaGamma ile CLTV
#####################################################

cltv_6 = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency'],
                                   cltv['T'],
                                   cltv['monetary'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_6.head()

cltv_6 = cltv_6.reset_index()
cltv_6.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_6.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

##############################################
# Segmentlere Ayırma
##############################################

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

###################################
# 1 Aylık CLTV
###################################

cltv_1 = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency'],
                                   cltv['T'],
                                   cltv['monetary'],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)

cltv_1 = cltv_1.reset_index()

cltv_1.sort_values(by = "clv", ascending=False).head(10)

###################################
# 12 Aylık CLTV
###################################

cltv_12 = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency'],
                                   cltv['T'],
                                   cltv['monetary'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)

cltv_12 = cltv_12.reset_index()

cltv_12.sort_values(by = "clv", ascending=False).head(10)

########################################
# Veritabanına Gönderme
########################################

cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)

cltv_final.to_sql(name='Ozan_Danabaş', con=conn, if_exists='replace', index=False)