import os

class InputParser:
    def __init__(self, input_file_name, path=None):
        # Construct the full file path if a path is specified
        self.input_file_name = os.path.join(path, input_file_name) if path else input_file_name
        self.input_list = {}

        # Check if the file exists
        if not os.path.isfile(self.input_file_name):
            raise FileNotFoundError(
                f"\n\n"
                f"*****************************************************\n"
                f"Input file {self.input_file_name} not found!\n"
                f"Terminating the program now\n"
                f"*****************************************************\n\n"
            )
        
        # Read and parse the file
        self.read_input_file()

    def read_input_file(self):
        """Parses the input file and populates the input_list dictionary."""
        with open(self.input_file_name, 'r') as file:
            for i_line, input_line in enumerate(file, start=1):
                # Strip whitespace
                input_line = input_line.strip()

                # Skip full-line comments and blank lines
                if not input_line or input_line.startswith('#'):
                    continue

                # Remove inline comments starting with '#'
                input_line = input_line.split('#', 1)[0].strip()
                
                # If there's an '=' sign, parse name=value pairs
                if '=' in input_line:
                    input_name, input_value = map(str.strip, input_line.split('=', 1))
                    self.input_list[input_name] = input_value
                else:
                    # If line does not define an input or a comment, raise error
                    raise ValueError(
                        f"\n\n"
                        f"*********************************************************************\n"
                        f"Error found in line {i_line} while parsing the input file {self.input_file_name}.\n"
                        f"Line {i_line} does not define either an input (does not contain an equal sign)\n"
                        f"nor a comment (does not contain a hashtag).\n"
                        f"Line content: '{input_line}'\n"
                        f"Please fix this in the input file.\n"
                        f"Terminating the program now.\n"
                        f"*********************************************************************\n\n"
                    )

    def retrieve_input(self, requested_element):
        """Retrieve a variable as a string from the input list."""
        requested_element = requested_element.strip()
        
        if requested_element not in self.input_list:
            raise KeyError(
                f"\n\n"
                f"************************************************************\n"
                f"Requested element '{requested_element}' not found in {self.input_file_name}.\n"
                f"Please add '{requested_element} = value' somewhere in {self.input_file_name}.\n"
                f"Terminating the program now.\n"
                f"************************************************************\n\n"
            )

        return self.input_list[requested_element]

    def retrieve_int(self, requested_element):
        """Retrieve a variable as an integer."""
        return int(self.retrieve_input(requested_element))

    def retrieve_float(self, requested_element):
        """Retrieve a variable as a float."""
        return float(self.retrieve_input(requested_element))

    def retrieve_bool(self, requested_element):
        """Retrieve a variable as a boolean."""
        return self.retrieve_input(requested_element).lower() == "true"

    def retrieve_string(self, requested_element):
        """Retrieve a variable as a string."""
        return self.retrieve_input(requested_element)