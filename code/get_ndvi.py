# Save as: get_ndvi.py
import ee
import pandas as pd


PROJECT = 'globxplorer'  # your actual Earth Engine project

# Initialize safely
try:
    ee.Initialize(project=PROJECT)
except Exception as e:
    print("Authenticating Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=PROJECT)



# Define Nal Sarovar (point center)
point = ee.Geometry.Point(72.05, 22.77)

# Load MODIS NDVI
collection = (ee.ImageCollection('MODIS/061/MOD13Q1')
              .filterDate('2010-01-01', '2024-12-31')
              .select('NDVI'))

# Extract mean NDVI at point
def extract_ndvi(img):
    mean = img.reduceRegion(ee.Reducer.mean(), point, 250).get('NDVI')
    date = img.date().format('YYYY-MM-dd')
    return ee.Feature(None, {'date': date, 'ndvi': mean})

ts = collection.map(extract_ndvi)
data = ts.getInfo()['features']

# Save CSV
df = pd.DataFrame([f['properties'] for f in data])
df['ndvi'] = df['ndvi'].astype(float) / 10000  # Scale
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.to_csv('data/ndvi_nal_sarovar.csv', index=False)
print("NDVI saved! Shape:", df.shape)