import os

def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

if __name__ == '__main__':
    theta0 = 0
    theta1 = 0

    if False:
        # Load theta0 and theta1 from a file
        pass

    print("Given a Milegae of a car, i'll predict the price of the car!!")
    mileage = float(input("Enter the mileage km of the car: "))
    esimated_price = predict_price(mileage, theta0, theta1)
    print(f"The estimated price of the car is: {esimated_price} km of mileage: {mileage}")