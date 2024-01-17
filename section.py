def find_value(input_text, key_value_dict):
    for key, value in key_value_dict.items():
        if key in input_text:
            return value
    return "Key not found"

# Example usage:
input_string = input("Enter a text: ")
key_value_dict = {'Abetment of suicide': 306,'murder': 375,'Grievous Hurt': 320,'Causing miscarraige': 312,'Rape': 375, 'Attempt to murder': 307, 'robbery': 420, 'Thug': 310, 'Hurt': 319, 'Force': 349, 'Assault': 351, 'Kidnapping': 359, 'Rape': 375,'Punishment for Rape': 375,'Unnatural offences': 377,'Abduction': 362,'Punishment': 311}  # Add more key-value pairs as needed

result = find_value(input_string, key_value_dict)
print("Result:", result)
