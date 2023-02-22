import regex as re

def hex_validator(colors : str | list):
 
    # Regex to check valid
    # hexadecimal color code.
	if isinstance(colors, str):
		colors = [colors]
	
	regex = "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"

	for color in colors:
	
		# Compile the ReGex
		p = re.compile(regex)
	
		# If the string is empty
		# return false
		if(color == None):
			return False
	
		# Return if the string
		# matched the ReGex
		if not (re.search(p, color)):
			return False

	return True 
