import numpy as np
def main():
    redshift = input("Emulate at Redshift ([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]): ")
    small_scale = input("Less velocity bins of Ly-a forest ([True, False]): ")
    n_testing = int(input("Extra testing data number: "))

    # Convert the input to numpy.float64
    try:
        redshift_float64 = np.float64(redshift)
    except ValueError:
        print("Invalid input. Please enter a valid redshift.")
    return redshift_float64, small_scale, n_testing

redshift, small_scale, n_testing = main()

#if __name__ == "__main__":

