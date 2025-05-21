def get_percentage_input(index):
    while True:
        try:
            percent = float(input(f"Enter probability {index+1} (%): "))
            if 0 <= percent <= 100:
                return percent
            else:
                print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    print("Enter 5 probabilities as percentages (0% to 100%).")
    probabilities = []

    for i in range(5):
        p = get_percentage_input(i)
        probabilities.append(p)

    # Calculate the average probability
    common_percentage = sum(probabilities) / len(probabilities)

    print("\n--- Result ---")
    print(f"The common probability (average) is: {common_percentage:.2f}%")

if __name__ == "__main__":
    main()
