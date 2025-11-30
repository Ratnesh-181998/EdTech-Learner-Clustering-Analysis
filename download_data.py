import requests

url = "https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/002/856/original/scaler_clustering.csv"
output_file = "scaler_clustering.csv"

response = requests.get(url)

with open(output_file, "wb") as f:
    f.write(response.content)

print("Download completed! File saved as", output_file)
