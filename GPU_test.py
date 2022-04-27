import pandas as pd

df = pd.DataFrame(data={"m":[0,1]})
print(df.to_numpy().astype(str))
print(df.to_numpy().transpose()[0].astype(str))
