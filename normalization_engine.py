from utils.normalization_functions import get_category, preprocess, entity_matcher, clustering_func
import time


def main():
    # Terminal messages
    print("\nWelcome! Here's how the engine works:\n")
    print("- Enter an entity that you would like to normalize and press Enter.\n")
    print("- Repeat until all of your entities have been entered.\n")
    print("- Type \"DONE\" when you're done entering the entities.\n")
    print("- Sit back and wait for the entities to be normalized and saved as a CSV file in your current directory.\n")

    # Create list to accept user input strings
    input_list = []
    while True:
        input_string = input("Enter next entity: ")
        if input_string == "DONE":
            break
        else:
            input_list.append(input_string)

    # Start timed clustering process
    print("\nEntity clustering will start now.\n")
    start = time.time()
    categorized_df = get_category(input_list)
    cleaned_df = preprocess(categorized_df)
    clustered_df = clustering_func(cleaned_df)

    # Save dataframe as CSV in current directory and print ending message
    filename = 'normalized_entities' + str(time.strftime("%Y%m%d-%H%M%S")) + '.csv'
    clustered_df.to_csv(filename, index = False)
    end = time.time()
    runtime = round((end - start), 2)
    print(f"Runtime of the program was {runtime} seconds.\n\n{filename} has been saved in the current directory.\n")

if __name__ == "__main__":
    main()
