# Databricks notebook source
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from ggplot import *
import warnings

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

#ETL process(Extract data from website into dbfs)
import urllib.request
urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", #"/tmp/sf_03_18.csv")
dbutils.fs.mv("file:/tmp/sf_03_18.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))

# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"
# use this file name later

# COMMAND ----------

# DBTITLE 1,Data Preprocessing 
# read data from the data storage
# please upload your data into databricks community at first. 
crime_data_lines = sc.textFile(data_path)

# COMMAND ----------

#prepare data 
df_crimes = crime_data_lines.map(lambda line: [x.strip('"') for x in next(reader([line]))])

# COMMAND ----------

#get header
header = df_crimes.first()
print(header)

# COMMAND ----------

#get the first line of data
#display(crimes.take(3))

#get the total number of data 
print(crimes.count())

# COMMAND ----------

##Spark data frame
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

##ETL transformation process (drop useless columns)
df_opt1 = df_opt1.drop(':@computed_region_yftq_j783', ':@computed_region_p5aj_wyqh', ':@computed_region_rxqg_mtj9', ':@computed_region_bh8s_q3mv', ':@computed_region_fyvs_ahh9', ':@computed_region_9dfj_4gjx', ':@computed_region_n4xg_c4py', ':@computed_region_4isq_27mq', ':@computed_region_fcz8_est8', ':@computed_region_pigm_ib2e', ':@computed_region_9jxd_iqea',
':@computed_region_6pnf_4xz7', ':@computed_region_6ezc_tdp2', ':@computed_region_h4ep_8xdi', ':@computed_region_nqbw_i6c3',
':@computed_region_2dwj_jsy4')

# COMMAND ----------

display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

####ETL transformation process (change string to date data type)
df_opt1 = spark.sql("SELECT IncidntNum, Category, Descript, DayOfWeek, to_date(Date, 'MM/dd/yyyy') AS Date, to_timestamp(Time, 'HH:mm') AS Time, PdDistrict, Resolution, Address, X, Y, Location, PdId FROM sf_crime")
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

##ETL transformation and loading process
##Change the data type of column 'X', 'Y'. Rename the name of column 'X' and 'Y'
df_opt1 =df_opt1.withColumn('X',df_opt1['X'].cast('float')).withColumn('Y',df_opt1['Y'].cast('float'))
df_opt1 = df_opt1.withColumnRenamed('X','Longitude').withColumnRenamed('Y','Latitude')
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

# inspect the schema of the data frame
df_opt1.printSchema()

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

# DBTITLE 1,Q1 counts the number of crimes for different category
crimeCategory = spark.sql("SELECT  category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC")
display(crimeCategory)

# COMMAND ----------

crimes_pd_df = crimeCategory.toPandas()

# COMMAND ----------

# DBTITLE 1,Q2 Counts the number of crimes for different district, and visualize your results
crimeDistrict = spark.sql("SELECT PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY PdDistrict ORDER BY Count DESC")
display(crimeDistrict)

# COMMAND ----------

# DBTITLE 1,Q3 Count the number of crimes each "Sunday" at "SF downtown".
##UDF
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf

def isDowntownBool(x,y):
  return True if 37.788566<float(y or 0)<37.799318 and -122.406513<float(x or 0)<-122.394881 else False #range
isDownTown = udf(lambda x,y: isDowntownBool(x,y), BooleanType())

# COMMAND ----------

SundayDT = spark.sql("SELECT Date, DayOfWeek, count(*) as Count from sf_crime \
                   WHERE DayOfWeek = 'Sunday' and \
                   pow(Latitude - 37.793942, 2) + pow(Longitude + 122.400697, 2) < pow(0.005, 2) \
                   GROUP BY Date, DayOfWeek ORDER BY Date")
display(SundayDT)

# COMMAND ----------

# DBTITLE 1,Q4 Analyze the number of crime in each month of 2015, 2016, 2017, 2018.


# COMMAND ----------

##create a new table(view)including only the information we need for this question
Month = df_opt1.select(year(df_opt1.Date).alias('Year'), month(df_opt1.Date).alias('Month'), dayofmonth(df_opt1.Date).alias('Day_of_Month'))
display(Month)
Month.createOrReplaceTempView("number_month")

# COMMAND ----------

##The number of crime in each month of 2015
Num2015 = spark.sql("SELECT Month, Count(*) AS Crime_num FROM number_month WHERE Year = 2015 GROUP BY Month ORDER BY Month")
display(Num2015)

# COMMAND ----------

##The number of crime in each month of 2016
Num2016 = spark.sql("SELECT Month, Count(*) AS Crime_num FROM number_month WHERE Year = 2016 GROUP BY Month ORDER BY Month")
display(Num2016)

# COMMAND ----------

##The number of crime in each month of 2017
Num2017 = spark.sql("SELECT Month, Count(*) AS Crime_num FROM number_month WHERE Year = 2017 GROUP BY Month ORDER BY Month")
display(Num2017)

# COMMAND ----------

##The number of crime in each month of 2018
Num2018 = spark.sql("SELECT Month, Count(*) AS Crime_num FROM number_month WHERE Year = 2018 GROUP BY Month ORDER BY Month")
display(Num2018)

# COMMAND ----------

# DBTITLE 1,Q5 
# MAGIC %md Analyze the number of crime w.r.t the hour in certain day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF.

# COMMAND ----------

# MAGIC %md Check the average number of crime on certain days like around Christmas and New Year during 2003-2018 and choose the date with the highest average number of crime. Then get the crime number of every single hour of that day.

# COMMAND ----------

#12/15
crimeBycertainDate = spark.sql("SELECT Distinct Date, COUNT(*) AS Count FROM sf_crime WHERE Date LIKE'%12/15/%' GROUP BY Date ORDER BY Date")
display(crimeBycertainDate)

# COMMAND ----------

crimeBycertainDate.createOrReplaceTempView("crimeBycertainDate")
Average = spark.sql("SELECT AVG(Count) AS Average FROM crimeBycertainDate")
display(Average)

# COMMAND ----------

#12/25
crimeBycertainDate1 = spark.sql("SELECT Distinct Date, COUNT(*) AS Count FROM sf_crime WHERE Date LIKE'%12/25/%' GROUP BY Date ORDER BY Date")
display(crimeBycertainDate1)

# COMMAND ----------

crimeBycertainDate1.createOrReplaceTempView("crimeBycertainDate1")
Average1 = spark.sql("SELECT AVG(Count) AS Average1 FROM crimeBycertainDate1")
display(Average1)

# COMMAND ----------

#12/24
crimeBycertainDate2 = spark.sql("SELECT Distinct Date, COUNT(*) AS Count FROM sf_crime WHERE Date LIKE'%12/24/%' GROUP BY Date ORDER BY Date")
display(crimeBycertainDate2)

# COMMAND ----------

crimeBycertainDate2.createOrReplaceTempView("crimeBycertainDate2")
Average2 = spark.sql("SELECT AVG(Count) AS Average2 FROM crimeBycertainDate2")
display(Average2)

# COMMAND ----------

12/26
crimeBycertainDate3 = spark.sql("SELECT Distinct Date, COUNT(*) AS Count FROM sf_crime WHERE Date LIKE'%12/26/%' GROUP BY Date ORDER BY Date")
display(crimeBycertainDate3)

# COMMAND ----------

crimeBycertainDate3.createOrReplaceTempView("crimeBycertainDate3")
Average3 = spark.sql("SELECT AVG(Count) AS Average3 FROM crimeBycertainDate3")
display(Average3)

# COMMAND ----------

12/31
crimeBycertainDate4 = spark.sql("SELECT Distinct Date, COUNT(*) AS Count FROM sf_crime WHERE Date LIKE'%12/31/%' GROUP BY Date ORDER BY Date")
display(crimeBycertainDate4)

# COMMAND ----------

crimeBycertainDate4.createOrReplaceTempView("crimeBycertainDate4")
Average4 = spark.sql("SELECT AVG(Count) AS Average4 FROM crimeBycertainDate4")
display(Average4)

# COMMAND ----------

#01/01
crimeBycertainDate5 = spark.sql("SELECT Distinct Date, COUNT(*) AS Count FROM sf_crime WHERE Date LIKE'%01/01/%' GROUP BY Date ORDER BY Date")
display(crimeBycertainDate5)

# COMMAND ----------

crimeBycertainDate5.createOrReplaceTempView("crimeBycertainDate5")
Average5 = spark.sql("SELECT AVG(Count) AS Average5 FROM crimeBycertainDate5")
display(Average5)

# COMMAND ----------

#01/02
crimeBycertainDate6 = spark.sql("SELECT Distinct Date, COUNT(*) AS Count FROM sf_crime WHERE Date LIKE'%01/02/%' GROUP BY Date ORDER BY Date")
display(crimeBycertainDate6)

# COMMAND ----------

crimeBycertainDate6.createOrReplaceTempView("crimeBycertainDate6")
Average6 = spark.sql("SELECT AVG(Count) AS Average6 FROM crimeBycertainDate6")
display(Average6)

# COMMAND ----------

# MAGIC %md From the statistics above, it is clear that the first day of New Year (January 1) got the largest average number of crime, while December 15 and December 31 ranked the second and third largest average number of crime. Take January 1 for analysis. Calculate the number of crime reported of every single hour in the New Year of 2016, 2017, and 2018.

# COMMAND ----------

Hour = df_opt1.select(year(df_opt1.Date).alias('Year'), month(df_opt1.Date).alias('Month'), dayofmonth(df_opt1.Date).alias('Day_of_Month'), hour(df_opt1.Time).alias('Hour'))
display(Hour)
Hour.createOrReplaceTempView("CertainDay")

# COMMAND ----------

##2016 New Year
NY2016 = spark.sql("SELECT Hour, Count(*) AS Crime_num FROM CertainDay WHERE Year = 2016 AND Month = 1 AND Day_of_Month = 1 GROUP BY Hour ORDER BY Hour")
display(NY2016)

# COMMAND ----------

##2017 New Year
NY2017 = spark.sql("SELECT Hour, Count(*) AS Crime_num FROM CertainDay WHERE Year = 2017 AND Month = 1 AND Day_of_Month = 1 GROUP BY Hour ORDER BY Hour")
display(NY2017)

# COMMAND ----------

##2018 New Year
NY2018 = spark.sql("SELECT Hour, Count(*) AS Crime_num FROM CertainDay WHERE Year = 2018 AND Month = 1 AND Day_of_Month = 1 GROUP BY Hour ORDER BY Hour")
display(NY2018)

# COMMAND ----------

# DBTITLE 1,Travel suggestions
# MAGIC %md From the result above, we can conclude that the dangerous time on NewYear is between midnight (0:00) and 1:00, and noon time (between 12:00 and 13:00) is also dangerous. We suggest travellers to avoid going out on New Year at midnight and noon(12:00 AM and 12:00 PM). In addition, travelers should pay more attention when going out on New Year's Eve (Dec.31) and Dec.15 

# COMMAND ----------

# DBTITLE 1,Q6
# MAGIC %md (1) Find out the top-3 dangerous district (2) Find out the crime event w.r.t category and time (hour) from the result of step 1 (3) give your advice to distribute the police based on your analysis results.

# COMMAND ----------

DistrictYr = df_opt1.select(year(df_opt1.Date).alias('Year'), df_opt1.columns[6]) 
display(DistrictYr)
DistrictYr.createOrReplaceTempView("DistrictYr")

# COMMAND ----------

crimeDistrictYr = spark.sql("SELECT Year, PdDistrict, COUNT(*) AS Count FROM DistrictYr GROUP BY PdDistrict, Year ORDER BY Count DESC")
display(crimeDistrictYr) 

# COMMAND ----------

# MAGIC %md From the result in Q2 and the result from above table, we can conclude that the top 3 dangerous district are: Southern, Mission, and Northern

# COMMAND ----------

##2.
CtgyT = df_opt1.select(year(df_opt1.Date).alias('Year'), hour(df_opt1.Time).alias('Hour'), df_opt1.columns[1], df_opt1.columns[6]) 
display(CtgyT)
CtgyT.createOrReplaceTempView("CtgyT")

# COMMAND ----------

southern = spark.sql("SELECT Hour, Category, COUNT(*) AS Count FROM CtgyT WHERE PdDistrict = 'SOUTHERN' GROUP BY Hour, Category ORDER BY Hour, Count DESC")
display(southern) 
southern.createOrReplaceTempView("southern")

# COMMAND ----------

southern_most = spark.sql("SELECT s.Hour, s.Category, s.Count FROM(SELECT Hour, MAX(Count) AS maxnum FROM southern GROUP BY Hour) AS x INNER JOIN southern AS s ON s.Hour = x.Hour AND s.Count = x.maxnum ORDER BY s.Hour")
display(southern_most)

# COMMAND ----------

mission = spark.sql("SELECT Hour, Category, COUNT(*) AS Count FROM CtgyT WHERE PdDistrict = 'MISSION' GROUP BY Hour, Category ORDER BY Hour, Count DESC")
display(mission) 
mission.createOrReplaceTempView("mission")

# COMMAND ----------

mission_most = spark.sql("SELECT m.Hour, m.Category, m.Count FROM(SELECT Hour, MAX(Count) AS maxnum FROM mission GROUP BY Hour) AS x INNER JOIN mission AS m ON m.Hour = x.Hour AND m.Count = x.maxnum ORDER BY m.Hour")
display(mission_most)

# COMMAND ----------

northern = spark.sql("SELECT Hour, Category, COUNT(*) AS Count FROM CtgyT WHERE PdDistrict = 'NORTHERN' GROUP BY Hour, Category ORDER BY Hour, Count DESC")
display(northern) 
northern.createOrReplaceTempView("northern")

# COMMAND ----------

northern_most = spark.sql("SELECT n.Hour, n.Category, n.Count FROM(SELECT Hour, MAX(Count) AS maxnum FROM northern GROUP BY Hour) AS x INNER JOIN northern AS n ON n.Hour = x.Hour AND n.Count = x.maxnum ORDER BY n.Hour")
display(northern_most)

# COMMAND ----------

# DBTITLE 1,Advice to distribute the police
# MAGIC %md Based on the statistics above, we can conclude that the police needs to be distributed more at Southern from 12:00 to 23:00, Mission from 12:00 to 13:00 and 18:00 to midnight, Northern from 17:00 to 22:00

# COMMAND ----------

# DBTITLE 1,Q7
# MAGIC %md For different category of crime, find the percentage of resolution. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

category = spark.sql("SELECT Category, Resolution, COUNT(*) AS Number FROM sf_crime GROUP BY Category, Resolution ORDER BY Number DESC")
display(category)

# COMMAND ----------

# MAGIC %md Larceny/Theft crime without any resolution and other kind of crime without any resolution are the two main crimes.  For larceny/theft crime, the policy depart should pay more attention and take action to them like arresting the criminals.

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md 
# MAGIC 1. In this project I aim to do a concrete analysis on San Francisco crime dataset containing the data from 2003 to May 2018, and get useful and insighful information then provide for the government to make better decisions and travelers to make better travel plan. 
# MAGIC 2. Technically, I analyzed the data using Spark Python. Fisrt, I did data preprocessing (data cleaning) using Spark SQL to delete useless columns and changed the data type for certain columns. Then I loaded the cleaned and ready-to-use data to a python dataframe, also generated a view (table) for SQL query. I accomplished OLAP by executing sql queries, then automatically generated plots to visualize the data.
# MAGIC 3. The results are clearly presented by tables and charts. I got multiple conclusions about the category, district, and time of crimes and gave advices and suggestions to the government and travelers.
# MAGIC 4. Due to the time and technical skills limitation, the San Francisco crime data analysis is not perfect. Further works like time series analysis will be done in the future. Hopefully more useful and exciting information will be found from the dataset.
