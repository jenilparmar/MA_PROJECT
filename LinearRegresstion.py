import numpy as np
import pandas as pd
data = pd.read_csv("data\\sh1.csv")
data.dropna(inplace=True)
#Matrix B
temperature = data['Temperature'].values   
#Matrix A
humidity_pressure = data[['Humidity', 'Pressure']].values    
#matrix A^T
transposed_humidity_pressure_matrix = np.transpose(humidity_pressure)   
#Y = alfa*(humidity) + bita*(pressure)
# res1 = (A^T) * (A) 
res1 = np.dot(transposed_humidity_pressure_matrix, humidity_pressure)
# res2 = (A^T) * (B) 
res2 = np.dot(transposed_humidity_pressure_matrix, temperature)
#inverse_matrix = ((A^T)(A))^-1
inverse_matrix = np.linalg.inv(res1)
#l = (inverse_matrix)*((A^T)(B))  // [alfa , bita]
l = np.dot(inverse_matrix, res2)
alfa  = l[0]  
bita  = l[1]
def predict(humidity , pressure):
    # Using ---> Y = alfa*(humidity) + bita*(pressure) 
    return (alfa*humidity + bita*pressure)

print(predict(53,1010.5))   #Output 19.487475995021555   original = 17.5   91
print(predict(71,1007.7))   #Output 16.51347314183274    original = 16.9   2
print(predict(57,1019.2))   #Output 19.085041170836057    original = 20.3   367
 