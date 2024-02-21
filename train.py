
from srcs.utils import split_data, read_data_source
from srcs.data import save_params, scale_data
from srcs.Ft_LinearRegression import Ft_LinearRegression
from sklearn.preprocessing import StandardScaler

input_data = "./data.csv"
model_file = "./model_params.pkl"

if __name__ == "__main__": 
    data = read_data_source(input_data)
    scaler = StandardScaler()  

    data_scaled = scale_data(data, scaler)
    (train_x, train_y), (valid_x, valid_y) = split_data(data_scaled)


    ft_lr = Ft_LinearRegression(data, 1e-4, 16000, 1000)
    params = ft_lr.train_model(train_x, train_y, valid_x, valid_y)

    mean, std = data.mean(), data.std()
    data = (data - mean) / std

    (train_x, train_y), (valid_x, valid_y) = split_data(data)

    ft_lr.metrics(scaler, params, train_x, train_y)
    save_params(params, model_file)