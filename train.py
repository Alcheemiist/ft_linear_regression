
from srcs.utils import split_data, read_data_source
from srcs.data import save_params, scale_data
from srcs.Ft_LinearRegression import Ft_LinearRegression
from sklearn.preprocessing import StandardScaler

input_data = "./data.csv"
model_file = "./model_params.pkl"

if __name__ == "__main__": 
    data = read_data_source(input_data)
    data_scaled = scale_data(data)
    (train_x, train_y), (valid_x, valid_y) = split_data(data_scaled)
    ft_lr = Ft_LinearRegression(data, 1e-5, 2000000, 100000)
    params = ft_lr.train_model(train_x, train_y, valid_x, valid_y)
    data_mean, data_std = data.mean(), data.std()
    ft_lr.metrics(params, train_x, train_y)
    save_params(data_mean, data_std, params, model_file)