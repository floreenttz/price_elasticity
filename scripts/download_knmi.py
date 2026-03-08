import os
import time
import urllib.request, urllib.parse
import pandas as pd
import ssl
import datetime
import boto3
from io import BytesIO

s3_client = boto3.client('s3')

def read_parquet_from_s3(bucket_name, s3_key):
    obj = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    return pd.read_parquet(BytesIO(obj['Body'].read()))

def write_parquet_to_s3(df, bucket_name, s3_key):
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, bucket_name, s3_key)

def KNMIWeather():
    """Function to get weather data from KNMI API and store it in a parquet file
    File is obtained from https://www.daggegevens.knmi.nl/klimatologie/uurgegevens
    A check is first performed if the file already exists, and if so, the startdate is set to the last date in the file,
    if not, the file is generated from scratch.

    Params: None

    Returns: enddate: datetime.date
    """

    bucket_name = 'prime-rel-ml'
    outputfilename = "raw_weather"
    s3_key = f"data-analytics/price-elasticity/data/{outputfilename}.parquet"

    # This context is needed, because KNMI uses HTTPS, but the certificate cannot be verified
    # They recommend not verifying the certificate, made explicit by the ssl context below.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        historicdf = read_parquet_from_s3(bucket_name, s3_key)
        startdate = historicdf.date.max().date()
        print(f"Start:{startdate}")
        enddate = datetime.date.today()
        print(f"End:{enddate}")
        startdate = startdate.strftime("%Y%m%d01")
        enddate = enddate.strftime("%Y%m%d24")
    except:
        print("Historicdf not found, generating from scratch")
        startdate = datetime.date(2000, 1, 1).strftime("%Y%m%d01")
        enddate = datetime.date(2024, 1, 1).strftime("%Y%m%d24")

    knmiurl = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
    # Requestdata specifies the properties we ask for from KNMI, stns are the stations by number
    # fmt means we want the data formatted as JSON (possible csv, json, and xml)
    stations = "260"  # De Bilt station

    requestdata = {"start": startdate, "end": enddate, "fmt": "json", "stns": stations}  # , "vars":"T:T10N:TD"}
    # As it is a post request we have to encode the request properly
    postdata = urllib.parse.urlencode(requestdata).encode()
    postreq = urllib.request.Request(knmiurl, data=postdata, method="POST", headers={'User-Agent': 'Mozilla/5.0'})

    with urllib.request.urlopen(postreq, context=ctx) as url:
        rawapidata = url.read().decode()  # Get raw information from API
        df = pd.read_json(rawapidata)
        df["date"] = pd.to_datetime(df["date"])
        
        try:
            historicdf = read_parquet_from_s3(bucket_name, s3_key)
            outputhistoricdf = pd.concat([historicdf, df], ignore_index=True)
            historicdf = None  # Saves memory, as interpreter might retain file in memory.
            print(f"Historic information found, appending new information up to {enddate}")
        except:
            print(f"Saved file invalid or non-existent, generating new one starting from {startdate} \n")
            outputhistoricdf = df

    outputhistoricdf = outputhistoricdf.sort_values(by=["date"]).drop_duplicates()
    write_parquet_to_s3(outputhistoricdf, bucket_name, s3_key)
    
    # Cleaning data for use
    outputhistoricdf = outputhistoricdf[['date', 'TG', 'TN', 'TX', 'RH', 'DR', 'SP']]
    outputhistoricdf = outputhistoricdf.rename(columns={'TG': 'mean_temp', 'TN': 'min_temp',
                                                        'TX': 'max_temp', 'RH': 'sum_rain',
                                                        'DR': 'len_rain', 'SP': 'perc_sun'})
    
    
    outputhistoricdf['mean_temp'] = outputhistoricdf['mean_temp'] / 10
    outputhistoricdf['max_temp'] = outputhistoricdf['max_temp'] / 10
    outputhistoricdf['min_temp'] = outputhistoricdf['min_temp'] / 10
    
    # Mean temp over last years
    outputhistoricdf['month'] = outputhistoricdf.date.dt.month
    outputhistoricdf['day'] = outputhistoricdf.date.dt.day
    mean_temps = outputhistoricdf.groupby(['month', 'day']).mean_temp.mean().reset_index()
    outputhistoricdf = outputhistoricdf.merge(mean_temps, on=['month', 'day'], suffixes=('', '_over_years'))

    outputhistoricdf['date'] = outputhistoricdf['date'].dt.date
    outputhistoricdf = outputhistoricdf.drop(columns=['month', 'day'])

    clean_s3_key = "data-analytics/price-elasticity/data/clean_weather.parquet"
    write_parquet_to_s3(outputhistoricdf, bucket_name, clean_s3_key)

    return enddate

def UpdateWeather():
    today = datetime.date.today().strftime("%Y%m%d0")
    while(KNMIWeather() < today):
        time.sleep(1)