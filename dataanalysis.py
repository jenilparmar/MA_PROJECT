import csv

# Function to read data from existing CSV file and write selected variables to a new CSV file
def create_new_csv(input_filename, output_filename):
    # List of variables to extract
    variables_to_extract = ['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1', 
                            'maxdewptm_1', 'mindewptm_1', 'maxpressurem_1', 'minpressurem_1', 
                            'precipm_1']

    # Open existing CSV file for reading
    with open(input_filename, 'r') as infile:
        reader = csv.DictReader(infile)
        
        # Open new CSV file for writing
        with open(output_filename, 'w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=variables_to_extract)
            
            # Write header
            writer.writeheader()
            
            # Iterate over rows in the existing CSV file
            for row in reader:
                # Create a dictionary containing only the selected variables
                selected_variables = {var: row[var] for var in variables_to_extract}
                
                # Write the selected variables to the new CSV file
                writer.writerow(selected_variables)

# Specify input and output filenames
input_filename = 'C:\\Users\\LEGION\\Desktop\\MA_PROJECT1\\MA_PROJECT\\data\\JaipurFinalCleanData _NextDay.csv'
output_filename = 'your_output_filename.csv'

# Call function to create new CSV file
create_new_csv(input_filename, output_filename)

print("New CSV file created successfully!")
